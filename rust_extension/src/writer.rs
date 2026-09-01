use serde::Serialize;
use rmp_serde::Serializer;


use crate::utils::TokenId;

use bytemuck::cast_slice;



#[derive(Serialize)]
struct Info {
    r#type: &'static str,
    global_engine_id: i32,
    group_id: i64,
    traj_id: i32
}

pub fn send_sequence_data(
    sender: &zmq::Socket,
    global_engine_id: i32,
    req_id: &str,
    allowed_tokens: Vec<Vec<TokenId>>,
    logprobs: Vec<f32>,
) {
    let mut parts = req_id.split('-');

    let group_id: i64 = parts
        .next()
        .ok_or("missing group_id").unwrap()
        .parse().unwrap();

    let traj_id: i32 = parts
        .next()
        .ok_or("missing traj_id").unwrap()
        .parse().unwrap();

    let info = Info {
        r#type: "rust",
        global_engine_id,
        group_id,
        traj_id,
    };

    let mut buf = Vec::new();
    info.serialize(
        &mut Serializer::new(&mut buf).with_struct_map()
    ).unwrap();

    let mut acc = 0i32;

    let allowed_tokens_offsets: Vec<i32> = std::iter::once(0)
        .chain(allowed_tokens.iter().map(|v| {
            acc += v.len() as i32;
            acc
        }))
        .collect();

    let allowed_tokens_flat: Vec<i32> = allowed_tokens.into_iter().flatten().collect();
    
    sender.send_multipart(vec![cast_slice(&buf), cast_slice(&logprobs), cast_slice(&allowed_tokens_flat), cast_slice(&allowed_tokens_offsets)], 0).unwrap();
}
