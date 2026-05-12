/**
 * Evo-Matrix Optimizer: 실시간 추천 및 Look-ahead 로직
 */

export let aiWeights = null;

export const loadAIWeights = async () => {
    try {
        const response = await fetch('./strategy_weights.json');
        if (!response.ok) throw new Error("File not found");
        const data = await response.json();
        aiWeights = data.weights;
        console.log("AI 가중치 로드 성공");
    } catch (error) {
        console.warn("AI 가중치 로드 실패 (휴리스틱 모드):", error);
    }
};

export const calculateDet = (m) =>
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6]);

export const calculateX = (matrices) => {
    let x = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
        x[i] = Math.floor(matrices.reduce((acc, m) => acc + m[i * 4], 0) / 6);
    }
    return x;
};

const axSum = (m, x) =>
    m[0]*x[0] + m[1]*x[1] + m[2]*x[2] +
    m[3]*x[0] + m[4]*x[1] + m[5]*x[2] +
    m[6]*x[0] + m[7]*x[1] + m[8]*x[2];

export const calculateWinProbability = (allMatrices, round) => {
    const x = calculateX(allMatrices);
    const scores = allMatrices.map(m => round <= 2 ? axSum(m, x) : calculateDet(m));
    const expScores = scores.map(s => Math.exp(s / 10));
    const total = expScores.reduce((a, b) => a + b, 0);
    return ((expScores[0] / total) * 100).toFixed(1);
};

/**
 * @param {number[]} myMatrix - 분석할 팀의 행렬
 * @param {number[][]} allMatrices - 전체 팀 행렬
 * @param {number} round - 현재 라운드
 * @param {number} teamIdx - myMatrix가 allMatrices에서 몇 번 팀인지 (기본값 0)
 */
export const getBestMoves = (myMatrix, allMatrices, round, teamIdx = 0) => {
    if (aiWeights) {
        const x = calculateX(allMatrices);
        const feat = new Array(20).fill(0);
        myMatrix.forEach((v, idx) => { feat[idx] = v; });
        x.forEach((v, idx) => { feat[9 + idx] = v; });
        feat[12] = round;

        const candidates = [];
        for (let actionIdx = 0; actionIdx < 18; actionIdx++) {
            let logit = 0;
            for (let f = 0; f < 20; f++) logit += feat[f] * aiWeights[f][actionIdx];

            const cellIdx = Math.floor(actionIdx / 2);
            const delta = (actionIdx % 2 === 0) ? 1 : -1;
            const nextMat = [...myMatrix];
            nextMat[cellIdx] += delta;

            candidates.push({
                index: cellIdx, delta,
                score: logit,
                det: calculateDet(nextMat),
                reason: logit > 0 ? "AI 추천 전략" : "점수 최적화",
                isUnique: false
            });
        }
        return candidates.sort((a, b) => b.score - a.score).slice(0, 3);
    }

    const prevX = calculateX(allMatrices);
    const candidates = [];

    for (let i = 0; i < 9; i++) {
        for (const delta of [-1, 1]) {
            const nextMat = [...myMatrix];
            nextMat[i] += delta;

            const nextAll = [...allMatrices];
            nextAll[teamIdx] = nextMat;
            const nextX = calculateX(nextAll);

            let score = round <= 2 ? axSum(nextMat, nextX) : calculateDet(nextMat);

            const otherScores = nextAll
                .filter((_, idx) => idx !== teamIdx)
                .map(m => round <= 2 ? axSum(m, nextX) : calculateDet(m));

            const isUniqueWinner = otherScores.every(s => s < score);
            const tieBreakerBonus = isUniqueWinner ? 50 : 0;

            const det = calculateDet(nextMat);
            if (round === 5 && det === 0) score = -5000;
            const singularityPenalty = det === 0 ? -1000 : 0;

            let reason = "점수 최적화";
            if (nextX[1] < prevX[1]) reason = "x2 벡터 하향을 통한 상대 견제";
            if (isUniqueWinner) reason = "단독 1위 확보";

            candidates.push({
                index: i, delta,
                score: score + singularityPenalty + tieBreakerBonus,
                det, xImpact: nextX, isUnique: isUniqueWinner, reason
            });
        }
    }

    return candidates.sort((a, b) => b.score - a.score).slice(0, 3);
};

export const getPrivilegeMoves = (allMatrices, round) => {
    let bestPriv = null;
    let maxAdvantage = -Infinity;

    for (let t = 0; t < 6; t++) {
        const m = allMatrices[t];
        const swapped = [m[3], m[4], m[5], m[0], m[1], m[2], m[6], m[7], m[8]];
        const nextAll = [...allMatrices];
        nextAll[t] = swapped;
        const myProb = parseFloat(calculateWinProbability(nextAll, round));
        if (myProb > maxAdvantage) {
            maxAdvantage = myProb;
            bestPriv = { team: t + 1, action: "1행과 2행 교환", impact: myProb };
        }
    }
    return bestPriv;
};

export const performLookAhead = (myMatrix, round) => {
    const det = calculateDet(myMatrix);
    if (det === 0) return { warning: true, prob: 100, msg: "이미 특이 행렬입니다! 즉시 수정하세요." };

    const row1 = myMatrix.slice(0, 3);
    const row2 = myMatrix.slice(3, 6);
    const row3 = myMatrix.slice(6, 9);

    const isNearDependent = (r1, r2) => {
        if (r2.every(v => v === 0)) return false;
        const ratios = r1.map((v, i) => v / (r2[i] || 1));
        const avg = ratios.reduce((a, b) => a + b, 0) / 3;
        return ratios.every(r => Math.abs(r - avg) < 0.3);
    };

    if (isNearDependent(row1, row2) || isNearDependent(row2, row3) || isNearDependent(row1, row3)) {
        return { warning: true, prob: 75, msg: "행 사이의 독립성이 낮습니다. 비대각 성분을 수정하세요." };
    }
    return { warning: false, prob: 5, msg: "안정적인 가역 상태입니다." };
};
