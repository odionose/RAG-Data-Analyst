# Application Architecture

## The ETL Pipeline

![The ETL Pipeline](/_images/etl_pipeline.png)

The pipeline has one job: take raw PDF filings and turn them into searchable vectors in Qdrant, with structured numbers in SQLite on the side. It runs in 5 sequential stages:

- Input — The SEC 10-K filings saved as company_2025.pdf in `data/documents/`. The naming convention matters because the chunker parses company name and year directly from the filename to attach as metadata to every vector.

- Extract — `extract_pdfs.py` uses pypdf to read each PDF page by page and concatenate the text into a .txt file in `data/extracted/`. Some pages may come back blank if they're image-based, which is why the script filters out empty pages before joining.

- Clean — `clean.py` reads from `data/extracted/` and strips PDF-specific noise: broken hyphenation (busi-\ness → business), page numbers, running headers, table of contents dot leaders, and Unicode ligatures introduced by PDF fonts. Output goes to data/cleaned/.

- Chunk — `chunker.py` splits each cleaned document using RecursiveCharacterTextSplitter on paragraph boundaries, then checks cosine similarity between adjacent chunks. If two neighbouring chunks are more than 85% similar they get merged — this keeps related ideas together rather than splitting a paragraph mid-thought. Every chunk gets a metadata dict attached: company_name, document_year, source_file, chunk_index.

- Embed — `pipeline.py` loads `all-MiniLM-L6-v2` locally (no API call, no cost) and encodes every chunk into a `384-dimensional vector`. IDs are generated with hashlib.md5 based on filename + chunk index so re-running the pipeline overwrites existing vectors rather than duplicating them.

- Store — vectors with their metadata payloads upsert into Qdrant Cloud in batches of 100. In parallel, `extract_financials.py` sends each PDF's financial section to Gemini and parses the JSON response into financials.db — this is the only stage that uses an API call.

## The Agentic Graph

![The Agentic Graph](/_images/agent_graph.png)

The agent is a directed graph with two nodes that cycle until the LLM decides it has enough information to answer.

- Entry — the `FastAPI POST /query` endpoint wraps the user's query string in a HumanMessage and passes it into the graph as the initial state. The state is a typed dict containing the message history and a CostTracker instance.

- Agent node — `Gemini 2.5 Flash Lite` receives the full message history plus a system prompt describing both tools. It returns either a tool call (it needs more information) or a plain text response (it has enough to answer). Token counts are read from response.usage_metadata and recorded by CostTracker on every pass through this node.

- should_continue() — the router function checks the last message. If it contains tool_calls, the graph routes to the tools node. If not, it routes to END. This is the branching point of the ReAct pattern.
Tools node — LangGraph's built-in ToolNode receives the tool call and dispatches to the right function:

- execute_sql — builds a SQL query against financials.db, executes it with sqlite3, and returns formatted rows. If SQLite raises an error, the error message is returned as the observation so Gemini can correct its query on the next loop iteration — this is the error recovery loop.

- search_vector_db — first calls Instructor to extract a VectorSearchParams Pydantic model from the query (pulling out company_name and document_year if present), then encodes the semantic query with `all-MiniLM-L6-v2`, and runs a filtered search against `Qdrant Cloud`. The metadata filter narrows results to the right company and year before the semantic search runs.

- Observation — the tool result gets appended to the message history as a ToolMessage and the graph loops back to the agent node. Gemini reads the updated history — original query + tool call + observation — and decides whether to call another tool or answer.

- END — when Gemini returns a message with no tool calls, the graph exits. The final message content is the answer. CostTracker.summary() is called to produce the cost breakdown, and FastAPI returns the full QueryResponse with answer, cost_summary, and latency_seconds.

The dashed line in the diagram represents this loop — a single query might pass through the agent node 2–4 times depending on how many tools it needs to call.

See the [results.json](/eval/results.json) for the RAGAS Evaluation.


## RAGAS Evaluation Report

This evaluation is based on the RAGAS Evaluations results in [results.json](/eval/results.json)

| Metric           | Meaning                                                             |
| ---------------- | ------------------------------------------------------------------- |
| Faithfulness     | Whether the answer is grounded in the retrieved/contextual evidence |
| Answer Relevance | Whether the answer directly addresses the user’s query              |

| Metric                   |    Score |
| ------------------------ | -------: |
| Average Faithfulness     | **0.80** |
| Average Answer Relevance | **0.70** |
| Combined Average Score   | **0.75** |


### Per-Query RAGAS Evaluation

| Query ID | Query                                                                                     | Faithfulness | Answer Relevance | Evaluation              |
| -------: | ----------------------------------------------------------------------------------------- | -----------: | ---------------: | ----------------------- |
|        1 | What was Apple's total revenue in 2025?                                                   |          1.0 |              1.0 | Excellent               |
|        2 | What are the main risk factors Amazon mentions in their 2025 filing?                      |          1.0 |              0.5 | Faithful but incomplete |
|        3 | Compare Apple and Google's net income in 2025. Which was more profitable?                 |          1.0 |              1.0 | Excellent               |
|        4 | How does Amazon describe its cloud computing strategy in 2025?                            |          1.0 |              1.0 | Excellent               |
|        5 | Which of the 4 companies had the highest R&D spending as a percentage of revenue in 2025? |          0.0 |              0.0 | Failed                  |


## Cost Analysis

This is based on the cost of the RAGAS Evaluations results in [results.json](/eval/results.json)

Assumption: 1 GCP credit = $1.00 USD equivalent.

### Summary

| Metric                         |                Value |
| ------------------------------ | -------------------: |
| Total queries evaluated        |                    5 |
| Total loops                    |                   10 |
| Total GCP credits consumed     | **0.001001 credits** |
| Total cost                     |        **$0.001001** |
| Average cost per query         |       **$0.0002002** |
| Estimated cost per 100 queries |         **$0.02002** |

### Per-Query Cost Breakdown

| Query ID | Total Loops |  Cost USD | GCP Credits Consumed | Share of Total Cost | Cost per 100 Similar Queries |
| -------: | ----------: | --------: | -------------------: | ------------------: | ---------------------------: |
|        1 |           2 | $0.000119 |             0.000119 |              11.89% |                      $0.0119 |
|        2 |           2 | $0.000215 |             0.000215 |              21.48% |                      $0.0215 |
|        3 |           2 | $0.000138 |             0.000138 |              13.79% |                      $0.0138 |
|        4 |           3 | $0.000477 |             0.000477 |              47.65% |                      $0.0477 |
|        5 |           1 | $0.000052 |             0.000052 |               5.19% |                      $0.0052 |

### Formula Used

Total cost = sum(total_cost_usd)
           = 0.000119 + 0.000215 + 0.000138 + 0.000477 + 0.000052
           = $0.001001

Average cost per query = total cost / number of queries
                       = 0.001001 / 5
                       = $0.0002002

Cost per 100 queries = average cost per query × 100
                     = 0.0002002 × 100
                     = $0.02002
