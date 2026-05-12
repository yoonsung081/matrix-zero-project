import React, { useState, useMemo } from 'react';
import { calculateDet, calculateX, getBestMoves, performLookAhead } from '../logic/strategy_logic';

const EvoMatrixHelper = () => {
  const [matrices, setMatrices] = useState(Array(6).fill(Array(9).fill(0)));
  const [round, setRound] = useState(1);
  const [myTeam, setMyTeam] = useState(0);

  const currentX = useMemo(() => calculateX(matrices), [matrices]);
  const bestMoves = useMemo(() => getBestMoves(matrices[myTeam], matrices, round), [matrices, round, myTeam]);
  const lookAhead = useMemo(() => performLookAhead(matrices[myTeam], round), [matrices, myTeam, round]);

  const updateCell = (mIdx, cIdx, val) => {
    const newMatrices = matrices.map((m, i) => 
      i === mIdx ? m.map((c, j) => j === cIdx ? parseInt(val) || 0 : c) : m
    );
    setMatrices(newMatrices);
  };

  return (
    <div className="p-6 bg-gray-900 min-h-screen text-emerald-400 font-mono">
      <div className="flex justify-between items-center mb-8 border-b border-emerald-800 pb-4">
        <h1 className="text-2xl font-bold tracking-widest">EVO-MATRIX OPTIMIZER v2.1</h1>
        <div className="flex gap-4 items-center">
          <span className="text-gray-500">ROUND:</span>
          <select value={round} onChange={(e) => setRound(Number(e.target.value))} className="bg-black border border-emerald-500 p-1">
            {[1,2,3,4,5].map(r => <option key={r} value={r}>{r}</option>)}
          </select>
          <div className="bg-emerald-900 px-3 py-1 rounded">X-Vector: [{currentX.join(', ')}]</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {matrices.map((matrix, mIdx) => (
          <div key={mIdx} className={`p-4 border ${mIdx === myTeam ? 'border-emerald-400 bg-emerald-950/30' : 'border-gray-700'}`}>
            <div className="flex justify-between mb-2 text-xs">
              <span>TEAM {mIdx + 1} {mIdx === myTeam && "(MY)"}</span>
              <span>DET: {calculateDet(matrix)}</span>
            </div>
            <div className="grid grid-cols-3 gap-1">
              {matrix.map((cell, cIdx) => (
                <input
                  key={cIdx}
                  type="number"
                  value={cell}
                  onChange={(e) => updateCell(mIdx, cIdx, e.target.value)}
                  className={`w-full h-12 text-center bg-black border ${
                    bestMoves.some(m => m.index === cIdx && mIdx === myTeam) ? 'border-yellow-400 animate-pulse' : 'border-gray-600'
                  } focus:outline-none focus:border-emerald-400`}
                  placeholder={mIdx !== myTeam && (cIdx === 0 || cIdx === 4 || cIdx === 8) ? "*" : ""}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* 전략 가이드 */}
        <div className="bg-black/50 p-6 border border-emerald-500/30 rounded-lg">
          <h2 className="text-xl font-bold mb-4 text-emerald-200">🚀 Best Move Recommendations</h2>
          <div className="space-y-4">
            {bestMoves.map((move, i) => (
              <div key={i} className="flex justify-between items-center p-3 bg-gray-800 rounded border-l-4 border-yellow-500">
                <div>
                  <span className="text-yellow-500 font-bold">TOP {i+1}:</span> 칸 {move.index}번을 {move.delta > 0 ? '+1' : '-1'}
                  <div className="text-xs text-gray-400 mt-1">예상 X 변화: [{move.xImpact.join(', ')}]</div>
                </div>
                <div className="text-right">
                  <div className="text-sm">Score: {move.score.toFixed(1)}</div>
                  <div className="text-xs text-gray-500">Det: {move.det}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 위험 분석 */}
        <div className="bg-black/50 p-6 border border-red-500/30 rounded-lg">
          <h2 className="text-xl font-bold mb-4 text-red-400">⚠️ Risk Analysis (Look-ahead)</h2>
          <div className="flex flex-col items-center justify-center h-32">
            <div className={`text-4xl font-black ${lookAhead.warning ? 'text-red-500 animate-bounce' : 'text-emerald-500'}`}>
              {lookAhead.prob}%
            </div>
            <div className="text-sm mt-2 text-gray-400">Singularity Probability at Round 5</div>
            {lookAhead.warning && (
              <div className="mt-4 px-4 py-2 bg-red-900/50 border border-red-500 text-red-200 text-xs text-center">
                위험: 행렬이 종속적으로 변하고 있습니다. <br /> 대각 성분을 조정하여 독립성을 확보하세요.
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="mt-8 text-center text-xs text-gray-600">
        Privilege Tip: 승리 시 상대의 행/열을 교환하여 상대 행렬식을 0으로 만들거나 내 대각선을 강화하세요.
      </div>
    </div>
  );
};

export default EvoMatrixHelper;