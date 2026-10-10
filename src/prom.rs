pub struct Sample {
    pub name: String,
    pub labels: Vec<(String, String)>,
    pub value: f64,
}

impl Sample {
    pub fn label(&self, key: &str) -> Option<&str> {
        self.labels
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

pub fn parse(text: &str) -> Vec<Sample> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((value_idx, value)) = split_value(&tokens) else {
            continue;
        };
        let left = tokens[..value_idx].join(" ");
        let (mut name, labels) = split_name_labels(&left);
        name = name.replace(':', "_");
        out.push(Sample { name, labels, value });
    }
    out
}

fn split_value(tokens: &[&str]) -> Option<(usize, f64)> {
    let mut depth: i32 = 0;
    let mut seen_brace = false;
    let mut name_end = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        if t.contains('{') {
            seen_brace = true;
        }
        depth += t.matches('{').count() as i32;
        depth -= t.matches('}').count() as i32;
        if !seen_brace || depth <= 0 {
            name_end = i + 1;
            break;
        }
    }
    let value = tokens.get(name_end)?.parse::<f64>().ok()?;
    Some((name_end, value))
}

fn split_name_labels(s: &str) -> (String, Vec<(String, String)>) {
    match s.find('{') {
        None => (s.to_string(), Vec::new()),
        Some(i) => {
            let name = s[..i].to_string();
            let end = s.rfind('}').unwrap_or(s.len());
            (name, parse_labels(&s[i + 1..end]))
        }
    }
}

fn parse_labels(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    loop {
        while chars.peek().is_some_and(|c| *c == ',' || c.is_whitespace()) {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        let mut key = String::new();
        let mut found_eq = false;
        while let Some(&c) = chars.peek() {
            if c == '=' {
                chars.next();
                found_eq = true;
                break;
            }
            key.push(c);
            chars.next();
        }
        if !found_eq || chars.peek() != Some(&'"') {
            continue;
        }
        chars.next();
        let mut val = String::new();
        loop {
            match chars.next() {
                Some('\\') => match chars.next() {
                    Some('n') => val.push('\n'),
                    Some(c) => val.push(c),
                    None => break,
                },
                Some('"') => break,
                Some(c) => val.push(c),
                None => break,
            }
        }
        out.push((key, val));
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub struct HistStats {
    pub count: f64,
    pub sum: f64,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
}

impl HistStats {
    pub fn avg(&self) -> Option<f64> {
        (self.count > 0.0).then(|| self.sum / self.count)
    }
}

pub fn histogram_quantile(q: f64, buckets: &[(f64, f64)]) -> Option<f64> {
    let total = buckets.last()?.1;
    if total <= 0.0 {
        return None;
    }
    let target = q * total;
    let mut prev_le = 0.0;
    let mut prev_cnt = 0.0;
    for &(le, cnt) in buckets {
        if cnt >= target {
            if le.is_infinite() {
                return Some(prev_le);
            }
            let span = cnt - prev_cnt;
            if span <= 0.0 {
                return Some(le);
            }
            return Some(prev_le + (target - prev_cnt) / span * (le - prev_le));
        }
        prev_le = le;
        prev_cnt = cnt;
    }
    Some(prev_le)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_and_skips_comments() {
        let text = "# HELP vllm:prompt_tokens_total x\n# TYPE vllm:prompt_tokens_total counter\nvllm:prompt_tokens_total 1234.0\n\nvllm:num_requests_running 3\n";
        let samples = parse(text);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].name, "vllm_prompt_tokens_total");
        assert_eq!(samples[0].value, 1234.0);
        assert_eq!(samples[1].value, 3.0);
    }

    #[test]
    fn parses_labels_with_quotes_and_escapes() {
        let text = "vllm:foo{model_name=\"a,b\\\"c\", le=\"0.5\"} 7\n";
        let samples = parse(text);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].label("model_name"), Some("a,b\"c"));
        assert_eq!(samples[0].label("le"), Some("0.5"));
        assert_eq!(samples[0].value, 7.0);
    }

    #[test]
    fn handles_timestamp_and_spaces_in_labels() {
        let text = "vllm:foo{model_name=\"llama 3\"} 1.5 1700000000000\nvllm:bar 2.5\n";
        let samples = parse(text);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].label("model_name"), Some("llama 3"));
        assert_eq!(samples[0].value, 1.5);
        assert_eq!(samples[1].value, 2.5);
    }

    #[test]
    fn skips_invalid_lines() {
        let text = "garbage\nvllm:x notanumber\nvllm:y 1.5\n";
        let samples = parse(text);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].name, "vllm_y");
    }

    #[test]
    fn quantile_interpolates() {
        let buckets = vec![(0.005, 1.0), (0.01, 2.0), (f64::INFINITY, 4.0)];
        let q = histogram_quantile(0.5, &buckets).unwrap();
        assert!((q - 0.01).abs() < 1e-9);
        let q = histogram_quantile(0.99, &buckets).unwrap();
        assert!((q - 0.01).abs() < 1e-9);
        assert!(histogram_quantile(0.5, &[(f64::INFINITY, 0.0)]).is_none());
    }

    #[test]
    fn quantile_below_first_bucket() {
        let buckets = vec![(0.01, 10.0), (f64::INFINITY, 10.0)];
        let q = histogram_quantile(0.5, &buckets).unwrap();
        assert!((q - 0.005).abs() < 1e-9);
    }
}
