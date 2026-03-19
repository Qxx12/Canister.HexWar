export { HighCommandAI } from './highCommand.ts'
export type {
  FrontStance,
  NeighborAssessment,
  GeopoliticalSnapshot,
  FrontDirective,
  StrategicPlan,
  TileConstraint,
} from './types.ts'
// Expose internals for testing and research
export { HistoryTracker } from './assessor/historyTracker.ts'
export { buildSnapshot } from './assessor/neighborAssessor.ts'
export { buildStrategicPlan } from './strategies/registry.ts'
export type { StrategyFn, PartialDirective } from './strategies/registry.ts'
