fn compute_half_window(view_dur_s: f64, sample_rate: f64) -> usize {
    ((view_dur_s / 2.0 + 2.5) * sample_rate) as usize
}

fn main() {
    let fs = 30000.0;
    let n_samples: usize = 69000000;
    
    // Jump to 1000s
    let view_start_s = 1000.0;
    let view_dur_s = 0.5;
    
    let view_first = (view_start_s * fs) as usize;
    let view_n = (view_dur_s * fs) as usize;
    let center = ((view_start_s + view_dur_s / 2.0) * fs) as usize;
    let half_window = compute_half_window(view_dur_s, fs);
    
    let first = center.saturating_sub(half_window);
    let n_samp = (half_window * 2).min(n_samples.saturating_sub(first));
    
    let buf_first_sample = first;
    let buf_n_samp = n_samp;
    
    let buf_end = buf_first_sample + buf_n_samp;
    let max_view_n = n_samples.saturating_sub(view_first);
    let expected_end = view_first + view_n.min(max_view_n);
    
    let m_view = buf_first_sample <= view_first && expected_end <= buf_end;
    
    println!("m_view: {}", m_view);
    println!("buf_first_sample: {} <= view_first: {}", buf_first_sample, view_first);
    println!("expected_end: {} <= buf_end: {}", expected_end, buf_end);
}
