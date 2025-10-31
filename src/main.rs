use std::env;
use std::io::{self, Read};
use std::path::Path;
use std::process;

use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base, p50k_base, p50k_edit, r50k_base};

const DEFAULT_ENCODING: &str = "o200k_base";

fn main() {
    if let Err(err) = run() {
        eprintln!("ttok-rs: {err}");
        process::exit(1);
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Count,
    Diff,
}

fn run() -> Result<(), String> {
    let mut args = env::args();
    let raw_program = args.next().unwrap_or_else(|| "ttok-rs".to_string());
    let program = display_name(&raw_program);
    let mut encoding = DEFAULT_ENCODING.to_string();
    let mut mode = Mode::Count;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-e" | "--encoding" => {
                let Some(value) = args.next() else {
                    return Err("missing value for --encoding".into());
                };
                encoding = value;
            }
            "-d" | "--diff" => {
                mode = Mode::Diff;
            }
            "-h" | "--help" => {
                print_help(&program);
                return Ok(());
            }
            "--list" => {
                print_supported();
                return Ok(());
            }
            other => {
                return Err(format!("unrecognized argument '{other}'"));
            }
        }
    }

    let mut buffer = Vec::new();
    io::stdin()
        .read_to_end(&mut buffer)
        .map_err(|err| format!("failed to read stdin: {err}"))?;

    let text = String::from_utf8_lossy(&buffer);
    let tokenizer = load_encoding(&encoding)?;

    match mode {
        Mode::Count => {
            let tokens = tokenizer.encode_with_special_tokens(&text);
            println!("{}", tokens.len());
        }
        Mode::Diff => {
            let (added, removed) = diff_token_totals(&tokenizer, &text);
            println!("{} {}", added, removed);
        }
    }

    Ok(())
}

fn load_encoding(name: &str) -> Result<CoreBPE, String> {
    match name {
        "o200k_base" => o200k_base().map_err(|err| err.to_string()),
        "cl100k_base" => cl100k_base().map_err(|err| err.to_string()),
        "p50k_base" => p50k_base().map_err(|err| err.to_string()),
        "p50k_edit" => p50k_edit().map_err(|err| err.to_string()),
        "r50k_base" | "gpt2" => r50k_base().map_err(|err| err.to_string()),
        other => Err(format!("unsupported encoding '{other}'")),
    }
}

fn print_help(program: &str) {
    println!("{program} — fast token counter using tiktoken-rs");
    println!();
    println!("Usage: {program} [OPTIONS] < input");
    println!();
    println!("Options:");
    let options = [
        (
            "-e, --encoding <name>",
            format!("Select tokenizer (default: {DEFAULT_ENCODING})"),
        ),
        (
            "-d, --diff",
            "Parse git diff and print added/removed token totals".to_string(),
        ),
        ("--list", "Show supported tokenizer names".to_string()),
        ("-h, --help", "Show this message".to_string()),
    ];
    for (flag, desc) in options {
        println!("  {:<22} {}", flag, desc);
    }
}

fn print_supported() {
    println!("Supported encodings:");
    println!("  o200k_base");
    println!("  cl100k_base");
    println!("  p50k_base");
    println!("  p50k_edit");
    println!("  r50k_base");
    println!("  gpt2");
}

fn display_name(raw: &str) -> String {
    Path::new(raw)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(raw)
        .to_string()
}

fn diff_token_totals(tokenizer: &CoreBPE, diff: &str) -> (usize, usize) {
    let mut added = 0usize;
    let mut removed = 0usize;

    for line in diff.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        if let Some(rest) = line.strip_prefix('+') {
            added += tokenizer.encode_with_special_tokens(rest).len();
        } else if let Some(rest) = line.strip_prefix('-') {
            removed += tokenizer.encode_with_special_tokens(rest).len();
        }
    }

    (added, removed)
}
