use std::{fs, process::Command};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_causal-ts"))
}

#[test]
fn full_cli_workflow() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path();
    let output = bin()
        .args([
            "generate",
            "--samples",
            "100",
            "--variables",
            "3",
            "--seed",
            "12",
            "--output",
        ])
        .arg(p.join("data.csv"))
        .arg("--truth")
        .arg(p.join("truth.json"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = bin()
        .args(["discover", "--input"])
        .arg(p.join("data.csv"))
        .args(["--threads", "2", "--test", "parcorr", "--output"])
        .arg(p.join("result.json"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let graph: serde_json::Value =
        serde_json::from_slice(&fs::read(p.join("result.json")).unwrap()).unwrap();
    assert_eq!(graph["algorithm"], "cedar");
    assert_eq!(graph["ci_test"], "parcorr");
    assert_eq!(graph["n_observations"], 100);
    assert_eq!(graph["var_names"].as_array().unwrap().len(), 3);
    let output = bin()
        .args(["evaluate", "--graph"])
        .arg(p.join("result.json"))
        .arg("--truth")
        .arg(p.join("truth.json"))
        .output()
        .unwrap();
    assert!(output.status.success());
    let metrics: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(metrics["f1"].is_number());
    let output = bin()
        .args(["ci-test", "--input"])
        .arg(p.join("data.csv"))
        .args(["--x", "X0", "--y", "X1", "--given", "X2"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let ci: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(ci["n_samples"], 100);
    for format in ["dot", "edges"] {
        let output = bin()
            .args(["discover", "--input"])
            .arg(p.join("data.csv"))
            .args(["--format", format])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8(output.stdout)
            .unwrap()
            .starts_with(if format == "dot" {
                "digraph"
            } else {
                "cause,effect"
            }));
    }
}

#[test]
fn bad_input_never_clobbers_existing_output() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("bad.csv");
    let output_path = tmp.path().join("result.json");
    fs::write(&input, "a,b\nhello,2\n").unwrap();
    fs::write(&output_path, "preserved").unwrap();
    let output = bin()
        .args(["discover", "--input"])
        .arg(input)
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("non-numeric"));
    assert_eq!(fs::read_to_string(output_path).unwrap(), "preserved");
}

#[test]
fn config_file_options_are_used_and_typos_fail() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path();
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/var.csv");
    fs::write(
        path.join("config.json"),
        "{\"max_lag\":2,\"include_autoreg\":false}",
    )
    .unwrap();
    let output = bin()
        .args(["discover", "--input"])
        .arg(&data)
        .arg("--config")
        .arg(path.join("config.json"))
        .output()
        .unwrap();
    assert!(output.status.success());
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["max_lag"], 2);
    assert!(result["edges"]
        .as_array()
        .unwrap()
        .iter()
        .all(|e| e["cause"] != e["effect"]));
    fs::write(path.join("config.json"), "{\"max_lags\":2}").unwrap();
    let output = bin()
        .args(["discover", "--input"])
        .arg(data)
        .arg("--config")
        .arg(path.join("config.json"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("unknown field"));
}

#[test]
fn unsupported_test_is_rejected() {
    let output = bin()
        .args(["discover", "--input", "unused.csv", "--test", "kci"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("invalid value"));
}

#[test]
fn cdnots_commands_preserve_endpoint_marks() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/var.csv");
    for algorithm in ["cdnots", "cdnots+"] {
        let out = bin()
            .args(["discover", "--algorithm", algorithm, "--input"])
            .arg(&data)
            .args(["--max-lag", "2", "--no-c"])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(value["algorithm"], algorithm);
        assert!(value["marks"].is_array());
        assert!(value["endpoint_matrix"].is_array());
        let out = bin()
            .args(["config", "--algorithm", algorithm])
            .output()
            .unwrap();
        assert!(out.status.success());
    }
}

#[cfg(feature = "neural")]
#[test]
fn grace_command_and_tensor_ci_execute() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/var.csv");
    let out = bin()
        .args(["discover", "--algorithm", "grace", "--input"])
        .arg(&data)
        .args([
            "--epochs",
            "1",
            "--no-c",
            "--backend",
            "cpu",
            "--max-lag",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["algorithm"], "grace");
    assert_eq!(v["backend"], "cpu");
    assert_eq!(v["training_runs"][0]["epochs"], 1);
    let out = bin()
        .args(["ci-test", "--input"])
        .arg(&data)
        .args(["--x", "X0", "--y", "X1", "--ci-backend", "cpu"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = bin()
        .args(["discover", "--algorithm", "cdnots", "--input"])
        .arg(data)
        .args(["--epochs", "1"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}
