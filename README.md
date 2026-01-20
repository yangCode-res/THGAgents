
## Features

- **Literature Review Fetching**: Automatically fetch and process biomedical literature reviews
- **Knowledge Graph Construction**: Build comprehensive knowledge graphs from text
- **Entity Extraction**: Identify biomedical entities (genes, proteins, diseases, drugs, etc.)
- **Relationship Extraction**: Extract relationships between entities
- **Hypothesis Generation**: Generate novel scientific hypotheses based on knowledge graphs
- **Multi-agent Architecture**: Modular agent system for different biomedical tasks



### API Keys Setup

Create a `.env` file in the project root directory:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE_URL=  # or your custom endpoint
OPENAI_MODEL=gpt-4o  # or gpt-4, gpt-3.5-turbo, etc.

# Optional: Mistral OCR API (for PDF processing)
MISTRAL_API_KEY=your_mistral_api_key_here
```

**Important**: Never commit your `.env` file or API keys to version control.

## Usage

### Run Full Benchmark Pipeline

Execute the main script to run the complete benchmark pipeline:

```bash
python run.py
```

This will:
1. Load benchmark data
2. Construct knowledge graphs from literature
3. Generate hypotheses
4. Save results to output directories

## Project Structure

```
HyAgent/
├── Agents/                 # Individual agent implementations
│   ├── Entity_extraction/     # Biomedical entity extraction
│   ├── Relationship_extraction/  # Entity relationship extraction
│   ├── HypothesisGenerationAgent/  # Hypothesis generation
│   └── ...
├── Core/                  # Core agent framework
├── Memory/                # Memory and data management
├── Logger/                # Logging utilities
├── utils/                 # Utility functions
├── new_benchMark/         # Benchmark evaluation framework
├── TypeDefinitions/       # Data type definitions
└── Config/                # Configuration files
```

## Key Components

### Agents
- **EntityExtractionAgent**: Extracts biomedical entities from text
- **RelationshipExtractionAgent**: Identifies relationships between entities
- **HypothesisGenerationAgent**: Generates scientific hypotheses
- **ReviewFetcherAgent**: Fetches relevant literature reviews

### Memory System
- **Subgraph**: Individual knowledge subgraphs
- **EntityStore**: Entity storage and deduplication
- **RelationStore**: Relationship storage
- **Memory**: Global memory management

## Output

Results are saved in the following structure:
```
new_benchMark/Group/{group_id}/
├── graph_register_snapshots/     # Initial graph construction
├── entity_extraction_snapshots/  # Entity extraction results
├── relationship_extraction_snapshots/  # Relationship extraction results
├── hypothesis_generation_snapshots/  # Generated hypotheses
└── evaluation_export/            # Evaluation results
```



