use clap::{Parser, Subcommand};
use notify::{EventKind, RecursiveMode, Watcher};
use pipe_graph::data::{Frame, FrameData, Payload};
use pipe_graph::exec::{Runtime, builtin_registry};
use pipe_graph::graph::{Graph, NodeId};
use std::fs;
use std::sync::mpsc;
use std::time::Duration;

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

        /// Watch the file for changes and automatically re-run the pipeline
        #[arg(short, long)]
        watch: bool,
    },
    /// Launch the Bevy visual editor for the pipeline
    Edit {
        /// Path to the pipeline file
        file: String,
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

fn run_pipeline(
    file: &str,
    input_node: &str,
    input_port: &str,
    output_node: &str,
    output_port: &str,
) {
    println!("\nLoading pipeline from {}...", file);
    let graph = match load_graph(file) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("❌ Failed to load graph: {}", e);
            return;
        }
    };
    
    let registry = builtin_registry();
    let mut runtime = match Runtime::instantiate(&graph, &registry) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ Graph compilation failed: {:?}", e);
            return;
        }
    };
    
    // Create a small 2x2 RGB test pattern
    let source = Frame::from_data(
        2,
        2,
        3,
        FrameData::U8(vec![1, 10, 100, 2, 20, 101, 3, 30, 102, 4, 40, 103]),
    );
    
    println!("Pushing test frame into {}.{}...", input_node, input_port);
    runtime.set_input(
        &NodeId(input_node.to_string()),
        input_port,
        Payload::Frame(source.clone()),
    );
    
    println!("Running pipeline...");
    if let Err(e) = runtime.run_once() {
        eprintln!("❌ Pipeline execution failed: {:?}", e);
        return;
    }
    
    println!("Reading output from {}.{}...", output_node, output_port);
    let output_payload = runtime.output(&NodeId(output_node.to_string()), output_port);
    
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
        }
    }
}

#[cfg(feature = "bevy")]
fn launch_editor(graph: Graph) {
    use bevy::prelude::*;
    use pipe_graph::systems::{PipeGraphEditorPlugin, GraphResource};

    println!("Launching Bevy editor...");
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(GraphResource(graph))
        .add_plugins(PipeGraphEditorPlugin)
        .run();
}

#[cfg(not(feature = "bevy"))]
fn launch_editor(_graph: Graph) {
    eprintln!("❌ Error: The editor requires the `bevy` feature. Re-run with `cargo run --features bevy -- edit <file>`.");
    std::process::exit(1);
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
        Commands::Run { file, input_node, input_port, output_node, output_port, watch } => {
            run_pipeline(file, input_node, input_port, output_node, output_port);
            
            if *watch {
                println!("\n👀 Watching {} for changes...", file);
                let (tx, rx) = mpsc::channel();
                let mut watcher = notify::recommended_watcher(tx).unwrap();
                
                let path = std::path::Path::new(file);
                watcher.watch(path, RecursiveMode::NonRecursive).unwrap();

                loop {
                    match rx.recv() {
                        Ok(Ok(event)) => {
                            // Only trigger on modify events
                            if !matches!(event.kind, EventKind::Access(_)) {
                                // Add a tiny debounce to let the file write finish
                                std::thread::sleep(Duration::from_millis(100));
                                run_pipeline(file, input_node, input_port, output_node, output_port);
                            }
                        }
                        Ok(Err(e)) => eprintln!("Watch error: {:?}", e),
                        Err(e) => {
                            eprintln!("Channel error: {:?}", e);
                            break;
                        }
                    }
                }
            }
        }
        Commands::Edit { file } => {
            println!("Loading pipeline from {}...", file);
            let graph = match load_graph(file) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Warning: failed to load graph ({}), starting with an empty graph.", e);
                    Graph::new()
                }
            };
            launch_editor(graph);
        }
    }
}
