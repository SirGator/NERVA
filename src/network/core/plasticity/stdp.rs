pub struct StdpParams {
    pub pre_trace_decay: f32, // decay rate of the presynaptic trace
    pub post_trace_decay: f32, // decay rate of the postsynaptic trace
    pub potentiation_rate: f32, // rate of potentiation 
    pub depression_rate: f32, // rate of depression 
}

pub struct StdpState {
    pub pre_trace: f32, // current value of the presynaptic trace
    pub post_trace: f32, // current value of the postsynaptic trace
}

