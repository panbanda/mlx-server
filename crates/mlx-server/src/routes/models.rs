use axum::{Json, extract::State};

use crate::{
    state::SharedState,
    types::openai::{ModelList, ModelObject},
};

pub async fn list_models(State(state): State<SharedState>) -> Json<ModelList> {
    let model_name = state.engine.model_name().to_owned();
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: model_name,
            object: "model",
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".to_owned(),
        }],
    })
}
