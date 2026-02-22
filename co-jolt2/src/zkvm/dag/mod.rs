pub mod coordinator;
pub mod stage;
pub mod state_manager;
pub mod worker;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Rep3DagStop {
    AfterCommitments,
    AfterStage1,
    AfterStage2,
    AfterStage3,
}
