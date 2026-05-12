/**
 * Evo-Matrix Optimizer: 실시간 추천 및 Look-ahead 로직
 */

export let aiWeights = null; // Python에서 학습된 가중치를 저장할 변수

export const loadAIWeights = async () => {
    try {
        const response = await fetch('./strategy_weights.json');
        if (!response.ok) throw new Error("File not found");
        const data = await response.json();
        aiWeights = data.weights;
        console.log("AI 가중치 로드 성공");
    } catch (error) {
        console.warn("AI 가중치 로드 실패 (휴리스틱 모드 작동):", error);
    }
};

export const calculateDet = (m) => {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
};

export const calculateX = (matrices) => {
    let x = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
        let sum = matrices.reduce((acc, m) => acc + m[i * 4], 0);
        x[i] = Math.floor(sum / 6);
    }
    return x;
};

export const calculateWinProbability = (allMatrices, round) => {
    // 간단한 Softmax 기반 승률 계산
    const scores = allMatrices.map((m, i) => {
        const x = calculateX(allMatrices);
        let ax_sum = 0;
        if (round <= 2) {
            ax_sum = m[0]*x[0] + m[1]*x[1] + m[2]*x[2] +
                     m[3]*x[0] + m[4]*x[1] + m[5]*x[2] +
                     m[6]*x[0] + m[7]*x[1] + m[8]*x[2];
        }
        return round <= 2 ? ax_sum : calculateDet(m); // 라운드 3-5는 행렬식 값 자체가 점수
    });
    const expScores = scores.map(s => Math.exp(s / 10));
    const total = expScores.reduce((a, b) => a + b, 0);
    return ((expScores[0] / total) * 100).toFixed(1);
};

export const getBestMoves = (myMatrix, allMatrices, round) => {
    // AI 가중치가 있고 AI 모드인 경우 (단순화를 위해 가중치가 있으면 AI 로직 우선 적용)
    if (aiWeights) {
        let candidates = [];
        // 특징 벡터 구성 (Python과 동일하게 20차원)
        const x = calculateX(allMatrices);
        const feat = new Array(20).fill(0);
        myMatrix.forEach((v, idx) => feat[idx] = v);
        x.forEach((v, idx) => feat[9 + idx] = v);
        feat[12] = round;

        // 모든 18가지 액션에 대해 Logit 계산
        for (let actionIdx = 0; actionIdx < 18; actionIdx++) {
            let logit = 0;
            for (let f = 0; f < 20; f++) {
                logit += feat[f] * aiWeights[f][actionIdx];
            }

            const cellIdx = Math.floor(actionIdx / 2);
            const delta = (actionIdx % 2 === 0) ? 1 : -1;
            
            // 시뮬레이션 점수 (UI 표시용)
            let nextMat = [...myMatrix];
            nextMat[cellIdx] += delta;
            const det = calculateDet(nextMat);

            candidates.push({
                index: cellIdx,
                delta: delta,
                score: logit, // AI의 판단 점수
                det: det,
                reason: logit > 0 ? "AI 추천 전략" : "점수 최적화",
                isUnique: false
            });
        }
        return candidates.sort((a, b) => b.score - a.score).slice(0, 3);
    }

    // 가중치가 없을 때 사용하는 기존 휴리스틱 로직
    const prevX = calculateX(allMatrices);
    let candidates = [];

    // 모든 가능한 이동 (9칸 * (+1/-1)) 시뮬레이션
    for (let i = 0; i < 9; i++) {
        for (let val of [-1, 1]) {
            let nextMat = [...myMatrix];
            nextMat[i] += val;
            
            // 내 행렬의 변화가 x에 미치는 영향 반영
            let nextAll = [...allMatrices];
            nextAll[0] = nextMat; // 가정: 내 팀이 0번 인덱스
            let nextX = calculateX(nextAll);
            
            // 라운드별 점수 계산
            let score = 0;
            if (round <= 2) {
                score = nextMat[0]*nextX[0] + nextMat[1]*nextX[1] + nextMat[2]*nextX[2] +
                        nextMat[3]*nextX[0] + nextMat[4]*nextX[1] + nextMat[5]*nextX[2] +
                        nextMat[6]*nextX[0] + nextMat[7]*nextX[1] + nextMat[8]*nextX[2];
            } else {
                score = calculateDet(nextMat); // Det는 값 자체가 점수
            }

            // 동점자 방지 및 승산(Odds) 가중치 (단독 1위 선호)
            let tieBreakerBonus = 0;
            const otherScores = nextAll.slice(1).map(m => {
                // Ax 합산 계산 로직 수정: 모든 성분 반영
                const otherAxSum = m[0]*nextX[0] + m[1]*nextX[1] + m[2]*nextX[2] + m[3]*nextX[0] + m[4]*nextX[1] + m[5]*nextX[2] + m[6]*nextX[0] + m[7]*nextX[1] + m[8]*nextX[2];
                return round <= 2 ? otherAxSum : calculateDet(m);            });
            const isUniqueWinner = otherScores.every(s => s < score);
            if (isUniqueWinner) tieBreakerBonus = 50; 

            // 가역성 체크 (싱귤래러티 방어)
            const det = calculateDet(nextMat);
            if (round === 5 && det === 0) score = -5000; // 5R에서 det 0은 즉시 패배
            const singularityPenalty = det === 0 ? -1000 : 0;

            // 전략적 코멘트 생성 (x 벡터 변화 분석)
            let reason = "점수 최적화";
            if (nextX[1] < prevX[1]) reason = "x2 벡터 하향을 통한 상대 견제";
            if (isUniqueWinner) reason = "단독 1위 확보";

            candidates.push({
                index: i,
                delta: val,
                score: score + singularityPenalty,
                det: det,
                xImpact: nextX,
                isUnique: isUniqueWinner,
                reason: reason
            });
        }
    }

    return candidates.sort((a, b) => b.score - a.score).slice(0, 3);
};

export const getPrivilegeMoves = (allMatrices, round) => {
    // 특권: 임의의 팀의 행/열 교환 시뮬레이션
    let bestPriv = null;
    let maxAdvantage = -Infinity;

    for (let t = 0; t < 6; t++) {
        let m = [...allMatrices[t]];
        // 행 교환 (1행-2행 교환 예시)
        let swapped = [m[3], m[4], m[5], m[0], m[1], m[2], m[6], m[7], m[8]];
        
        let nextAll = [...allMatrices];
        nextAll[t] = swapped;
        
        // 이 교환으로 인해 내가 1위가 되거나 공동 1위가 깨지는지 확인
        let myProb = parseFloat(calculateWinProbability(nextAll, round));
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

    // 단순화된 선형 종속성 체크
    const row1 = myMatrix.slice(0, 3);
    const row2 = myMatrix.slice(3, 6);
    const row3 = myMatrix.slice(6, 9);

    const isNearDependent = (r1, r2) => {
        if (r2.every(v => v === 0)) return false;
        const ratios = r1.map((v, i) => v / (r2[i] || 1));
        const avgRatio = ratios.reduce((a, b) => a + b, 0) / 3;
        return ratios.every(r => Math.abs(r - avgRatio) < 0.3);
    };

    if (isNearDependent(row1, row2) || isNearDependent(row2, row3) || isNearDependent(row1, row3)) {
        return { warning: true, prob: 75, msg: "행 사이의 독립성이 낮습니다. 비대각 성분을 수정하여 det=0을 방어하세요." };
    }
    return { warning: false, prob: 5, msg: "안정적인 가역 상태입니다." };
};