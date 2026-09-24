use clap::{Parser, Subcommand};
use pipe_graph::data::{Frame, FrameData, Payload};
use pipe_graph::exec::{Runtime, builtin_registry};
use pipe_graph::graph::{Graph, NodeId};
use std::fs;

#[derive(Parser)]
#[command(name = "pipe-graph")]
#[command(about = "Node graph editor runtime for data pipelines", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validates a pipeline definition file (TOML or YAML)
    Check {
        /// Path to the pipeline file
        file: String,
    },
    /// Executes a pipeline definition file
    Run {
        /// Path to the pipeline file
        file: String,
        
        /// Node ID to push the initial test frame into
        #[arg(long, default_value = "split")]
        input_node: String,
        
        /// Port ID to push the initial test frame into
        #[arg(long, default_value = "in")]
        input_port: String,
        
        /// Node ID to read the final output frame from
        #[arg(long, default_value = "merge")]
        output_node: String,
        
        /// Port ID to read the final output frame from
        #[arg(long, default_value = "out")]
        output_port: String,
    },
}

fn load_graph(path: &str) -> Result<Graph, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("Failed to read file {}: {}", path, e))?;
    if path.ends_with(".toml") {
        Graph::from_toml(&content).map_err(|e| e.to_string())
    } else if path.ends_with(".yaml") || path.ends_with(".yml") {
        Graph::from_yaml(&content).map_err(|e| e.to_string())
    } else {
        Err(format!("Unsupported file extension for {}", path))
    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Check { file } => {
            println!("Loading pipeline from {}...", file);
            match load_graph(file) {
                Ok(graph) => {
                    let registry = builtin_registry();
                    match Runtime::instantiate(&graph, &registry) {
                        Ok(_) => {
                            println!("✅ Graph is valid and compiled successfully!");
                            println!("  Nodes: {}", graph.nodes.len());
                            println!("  Edges: {}", graph.edges.len());
                        }
                        Err(e) => {
                            eprintln!("❌ Graph compilation failed: {:?}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ Failed to load graph: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Run { file, input_node, input_port, output_node, output_port } => {
            println!("Loading pipeline from {}...", file);
            let graph = load_graph(file).unwrap_or_else(|e| {
                eprintln!("❌ Failed to load graph: {}", e);
                std::process::exit(1);
            });
            
            let registry = builtin_registry();
            let mut runtime = Runtime::instantiate(&graph, &registry).unwrap_or_else(|e| {
                eprintln!("❌ Graph compilation failed: {:?}", e);
                std::process::exit(1);
            });
            
            // Create a small 2x2 RGB test pattern
            let source = Frame::from_data(
                2,
                2,
                3,
                FrameData::U8(vec![1, 10, 100, 2, 20, 101, 3, 30, 102, 4, 40, 103]),
            );
            
            println!("Pushing test frame into {}.{}...", input_node, input_port);
            runtime.set_input(
                &NodeId(input_node.clone()),
                input_port,
                Payload::Frame(source.clone()),
            );
            
            println!("Running pipeline...");
            runtime.run_once().unwrap_or_else(|e| {
                eprintln!("❌ Pipeline execution failed: {:?}", e);
                std::process::exit(1);
            });
            
            println!("Reading output from {}.{}...", output_node, output_port);
            let output_payload = runtime.output(&NodeId(output_node.clone()), output_port);
            
            match output_payload {
                Some(payload) => {
                    if let Some(frame) = payload.as_frame() {
                        println!("✅ Pipeline completed successfully!");
                        println!("  Output shape: {}x{}x{}", frame.width, frame.height, frame.channels);
                        println!("  Output data: {:?}", frame.data());
                    } else {
                        println!("✅ Pipeline completed, but output was not a Frame.");
                    }
                }
                None => {
                    eprintln!("❌ No output found at {}.{}", output_node, output_port);
                    std::process::exit(1);
                }
            }
        }
    }
}
