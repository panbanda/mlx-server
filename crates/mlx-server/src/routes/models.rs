use axum::{Json, extract::State};

use crate::{
    state::SharedState,
    types::openai::{ModelList, ModelObject},
};

pub async fn list_models(State(state): State<SharedState>) -> Json<ModelList> {
    let data: Vec<ModelObject> = state
        .engines
        .keys()
        .map(|name| ModelObject {
            id: name.clone(),
            object: "model",
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".to_owned(),
        })
        .collect();
    Json(ModelList { object: "list", data })
}
