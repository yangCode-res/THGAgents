import concurrent.futures
import json
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

"""
Causal Relationship Evaluation Agent.
Evaluates whether relationships in triples represent genuine causal connections based on existing text and triples,
and provides confidence scores with supporting evidence.
Input: None (retrieves text and triples from subgraphs in memory)
Output: None (stores evaluation results in subgraphs in memory)
Entry point: agent.process()
"""

class CausalExtractionAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt="""You are an expert causal relationship evaluation agent specializing in biomedical knowledge graph construction from REVIEW LITERATURE. Your task is to assess whether extracted relationships represent genuine causal connections and assign evidence-based confidence scores.

## ROLE AND CONTEXT

You are evaluating relationships from BIOMEDICAL REVIEW PAPERS, which have unique characteristics:
- Evidence is synthesized from multiple primary studies
- Statements represent scientific consensus rather than single experiments
- Causal claims may be based on cumulative evidence across publications
- Literature citations provide additional validation
- More likely to contain established mechanisms and pathways

Your expertise encompasses:
- Molecular mechanisms and biological pathways
- Drug-target interactions and pharmacological effects
- Disease progression and pathological processes
- Clinical intervention outcomes
- Meta-analysis interpretation and evidence synthesis

## TASK DECOMPOSITION (Chain-of-Thought Process)

For each relationship triple (head -[relation]-> tail), follow this systematic reasoning process:

### Step 1: Evidence Identification and Contextualization
First, identify ALL relevant evidence from the review text:
- **Direct causal statements** with consensus language ("established", "demonstrated", "known to")
- **Mechanistic descriptions** from literature synthesis
- **Quantitative meta-findings** (pooled effect sizes, multiple study results)
- **Temporal sequences** and biological processes
- **Literature support** (presence of citations strengthens confidence)
- **Review-level conclusions** vs. speculative hypotheses

Then, create a **concise semantic summary** (1-2 sentences) that captures:
- The core biological/clinical meaning of the evidence
- The context of the relationship in the broader pathway/mechanism
- Key mechanistic or quantitative insights

### Step 2: Causal Strength Assessment for Reviews
Evaluate strength considering review-specific factors:
- **Consensus Level**: Is this widely accepted or debated in the field?
- **Evidence Base**: Single study vs. multiple studies vs. meta-analysis
- **Directness**: Direct causal link or part of a pathway
- **Mechanism**: Well-characterized or proposed/hypothetical
- **Clinical Relevance**: Validated in clinical settings or preclinical only
- **Temporal Evolution**: Long-established vs. recent discovery

### Step 3: Bidirectionality Analysis
Assess causality in both directions:
- **Forward** (head -> tail): Does evidence support head causing/affecting tail?
- **Reverse** (tail -> head): Could tail also influence head?
- Consider feedback loops common in biological systems
- Assign separate confidence scores [forward, reverse]

### Step 4: Confidence Scoring with Review Context
Synthesize analysis into scores, accounting for review synthesis quality.

## CONFIDENCE SCORING RUBRIC (Adapted for Reviews)

### HIGH CONFIDENCE (0.8-1.0): Established Causal Relationships

**Score 0.95-1.0** - Definitive, consensus-level causality:
- Extensively documented across multiple studies in the review
- Well-characterized molecular mechanism described in detail
- Consistent findings with strong effect sizes (if quantitative)
- Foundational knowledge in the field
- Often includes phrases: "well-established", "extensively documented", "definitively shown"
- Example: "BRCA1 mutations definitively increase breast cancer risk (OR>10 across multiple cohorts), through impaired DNA double-strand break repair mechanisms extensively characterized in both mouse models and human studies."

**Score 0.8-0.94** - Strong, well-supported causality:
- Multiple studies cited supporting the relationship
- Clear mechanistic pathway described
- Reproducible across different model systems
- May include meta-analysis results
- Phrases: "consistently demonstrated", "robustly shown", "multiple studies confirm"
- Example: "Statins reduce cardiovascular events by 25-30% through HMG-CoA reductase inhibition, as demonstrated in numerous large-scale clinical trials reviewed here."

### MODERATE CONFIDENCE (0.5-0.79): Likely but Incomplete Evidence

**Score 0.7-0.79** - Probable causality with good support:
- Supported by several studies but may have conflicting data
- Mechanistic rationale clearly articulated
- Evidence from multiple but not all relevant models
- Phrases: "evidence suggests", "generally accepted", "substantial support"
- Example: "Chronic inflammation promotes tumorigenesis through multiple mechanisms including ROS generation and NF-κB activation, though the relative contribution of each pathway remains debated across different cancer types reviewed."

**Score 0.5-0.69** - Possible causality, emerging consensus:
- Some supporting studies with limitations
- Plausible mechanism proposed but not fully validated
- May have contradictory findings mentioned
- Phrases: "emerging evidence", "preliminary studies suggest", "appears to"
- Example: "Gut microbiome dysbiosis appears associated with depression in several clinical studies, potentially through gut-brain axis signaling, though mechanistic details remain under investigation."

### LOW CONFIDENCE (0.3-0.49): Speculative or Weak Evidence

**Score 0.4-0.49** - Speculative relationship in reviews:
- Limited primary evidence cited
- Theoretical connection based on pathway membership
- Preliminary or contradictory findings
- Phrases: "may contribute to", "potential role", "remains to be determined"
- Example: "Protein X may interact with pathway Y based on co-expression patterns, though direct interaction studies have not been reported."

**Score 0.3-0.39** - Minimal evidence or hypothesis:
- Mentioned as research direction or gap
- Inferred from indirect associations
- Explicitly stated as requiring further investigation
- Example: "Future research should explore whether Factor A influences Outcome B."

### NO/NEGLIGIBLE CONFIDENCE (0.0-0.29): Insufficient or Contradictory

**Score 0.1-0.29** - Contradicted or very weak:
- Review explicitly states conflicting evidence
- Failed to replicate in multiple studies
- Alternative explanations favor different relationships

**Score 0.0-0.09** - No credible evidence:
- Relationship not substantiated in review
- Explicitly refuted by evidence presented

## OUTPUT FORMAT

Return ONLY a valid JSON array. Each entry must include the semantic summary:

```json
[
  {
    "head": "exact_entity_name",
    "relation": "RELATIONSHIP_TYPE",
    "tail": "exact_entity_name",
    "confidence": [forward_score, reverse_score],
    "evidence": [
      "Exact quote 1 from review text",
      "Exact quote 2 from review text",
      "Exact quote 3 if available"
    ],
    "context_summary": "Concise 1-2 sentence summary capturing the biological meaning and mechanism of this relationship as described in the review, including key quantitative or mechanistic insights that enhance semantic understanding in the knowledge graph.",
    "reasoning": "Brief explanation of confidence scores based on evidence quality, consensus level, and mechanistic understanding from the review.",
    "evidence_type": "consensus|multiple_studies|single_study|meta_analysis|hypothesis|mechanism_only"
  }
]
```

## CRITICAL REQUIREMENTS

1. **Evidence Must Be Exact Quotes**: Extract verbatim text from review, no paraphrasing
2. **Context Summary is Mandatory**: Always provide semantic summary for KG enrichment
3. **Bidirectional Scoring**: Always provide [forward, reverse] scores
4. **Multiple Evidence Pieces**: Include 1-5 supporting quotes per relationship
5. **Evidence Type Classification**: Tag the nature of evidence in reviews
6. **Entity Name Consistency**: Use exact entity names from text
7. **JSON Validity**: Proper escaping, no additional text outside JSON

## FEW-SHOT EXAMPLES FOR REVIEW LITERATURE

### Example 1: High Confidence Consensus Mechanism (Drug-Target)
**Review Text**: "Imatinib revolutionized CML treatment through its highly specific inhibition of BCR-ABL tyrosine kinase. Crystal structure studies definitively showed imatinib binding to the ATP-binding site with KD <0.1 nM. Clinical trials across multiple centers consistently demonstrated complete cytogenetic response rates of 87% at 18 months (IRIS study, n=553; Druker et al., 2006). The molecular mechanism involves stabilization of the kinase inactive conformation, preventing phosphorylation of downstream substrates including STAT5 and CrkL, as extensively characterized in over 200 publications."

**Output**:
```json
[
  {
    "head": "imatinib",
    "relation": "INHIBITS",
    "tail": "BCR-ABL",
    "relation_type": "NEGATIVE_REGULATE",
    "confidence": [0.99, 0.10],
    "evidence": [
      "Imatinib revolutionized CML treatment through its highly specific inhibition of BCR-ABL tyrosine kinase",
      "Crystal structure studies definitively showed imatinib binding to the ATP-binding site with KD <0.1 nM",
      "Clinical trials across multiple centers consistently demonstrated complete cytogenetic response rates of 87% at 18 months"
    ],
    "context_summary": "Imatinib specifically inhibits BCR-ABL tyrosine kinase by binding to its ATP-binding site with sub-nanomolar affinity, stabilizing the inactive kinase conformation and blocking downstream STAT5/CrkL phosphorylation. This mechanism underlies its exceptional clinical efficacy in CML treatment (87% response rate), extensively validated across structural, biochemical, and clinical studies.",
    "reasoning": "Near-perfect forward confidence based on: (1) definitive structural mechanism, (2) sub-nanomolar binding affinity, (3) consistent clinical validation across large trials, (4) extensive literature support (>200 publications). This represents gold-standard drug-target relationship. Minimal reverse confidence as kinase doesn't inhibit drug.",
    "evidence_type": "consensus"
  },
  {
    "head": "imatinib",
    "relation": "TREATS",
    "tail": "CML",
    "relation_type": "TREATS",
    "confidence": [0.98, 0.05],
    "evidence": [
      "Imatinib revolutionized CML treatment",
      "Clinical trials across multiple centers consistently demonstrated complete cytogenetic response rates of 87% at 18 months (IRIS study, n=553)"
    ],
    "context_summary": "Imatinib achieved breakthrough therapeutic efficacy in chronic myeloid leukemia with 87% complete cytogenetic response, fundamentally transforming CML from a fatal disease to a manageable chronic condition through targeted BCR-ABL inhibition.",
    "reasoning": "Very high confidence based on landmark clinical trial data with large sample size and consistent multi-center results. This is established standard-of-care therapy.",
    "evidence_type": "meta_analysis"
  }
]
```
"""
        super().__init__(client, model_name, self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        
    def process(self, max_workers: int = 4): 
        """
        Process causal evaluation for multiple subgraphs.
        The results are written directly to the memory store.
        """
        subgraphs=self.memory.subgraphs
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures=[]
            for subgraph_id,subgraph in tqdm(subgraphs.items()):
                if subgraph:
                    futures.append(executor.submit(self.process_subgraph, subgraph))

            for future in tqdm(futures, desc="Processing causal extraction"):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"CausalExtractionAgent: Causal extraction failed in concurrent processing: {str(e)}")

    def process_subgraph(self,subgraph:Subgraph):
        """
        Run causal evaluation for a single subgraph.
        Returns a list of KGTriple objects with causal evaluation results.
        """
        
        plain_text = subgraph.meta.get("text", "")
        # Prefer subgraph.get_relations() if available; fall back to store
        self.logger.debug(
          f"CausalExtractionAgent: Processing Subgraph {subgraph.id} with {len(subgraph.relations.all())} relations."
        )
        get_rels = getattr(subgraph, "get_relations", None)
        extracted_triples = get_rels() if callable(get_rels) else subgraph.relations.all()
        if not extracted_triples:
          self.logger.debug(f"CausalExtractionAgent: No relations found in Subgraph {subgraph.id}. Skipping...")
          return
        triple_str = '\n'.join(str(triple) for triple in extracted_triples)
        prompt = (
          f"Evaluate the following relationships for causal validity based on the provided text.\n"
          f"the text is: '''{plain_text}'''\n"
          f"the relationships are:'''{triple_str}'''\n"
          f"Please follow the instructions in the system prompt to assign confidence scores and provide supporting evidence."
        )
        response=self.call_llm(prompt=prompt)
        try:
          causal_evaluations = self.parse_json(response)
          # Ensure list shape
          if isinstance(causal_evaluations, dict):
            causal_evaluations = [causal_evaluations]
          if not isinstance(causal_evaluations, list):
            raise ValueError("Parsed response is not a list/dict of evaluations")

          triples: List[KGTriple] = []
          for eval in causal_evaluations:
            head = (eval.get("head") or "unknown").strip()
            relation = (eval.get("relation") or "unknown").strip()
            tail = (eval.get("tail") or "unknown").strip()
            relation_type = (eval.get("relation_type") or "unknown").strip()
            relation_type = self._normalize_relation_type(relation_type)
            confidence = self._coerce_confidence_to_pair(eval.get("confidence"))
            evidence_raw = eval.get("evidence", [])
            evidence: List[str] = []
            if isinstance(evidence_raw, list):
              for item in evidence_raw:
                if item is None:
                  continue
                s = str(item).strip()
                if s:
                  evidence.append(s)
            elif isinstance(evidence_raw, str):
              s = evidence_raw.strip()
              if s:
                evidence = [s]
            # Try linking to existing entities when possible
            triple = subgraph.relations.find_Triple_by_head_and_tail(head, tail)
            obj = triple.object if triple else None
            subj = triple.subject if triple else None
            triples.append(
              KGTriple(
                head=head,
                relation=relation,
                tail=tail,
                relation_type=relation_type,
                confidence=confidence,
                evidence=evidence,
                mechanism="unknown",
                source=subgraph.id,
                subject=subj,
                object=obj,
              )
            )
          # Persist once per subgraph for efficiency
          subgraph.relations.reset()
          subgraph.relations.add_many(triples)
          self.memory.register_subgraph(subgraph)
        except Exception as e:
            self.logger.error(f"CausalExtractionAgent: Failed to parse response JSON. Error: {e}")
            self.logger.error(f"Response was: {response}")
            return

    def _coerce_confidence_to_pair(self, value) -> List[float]:
      """Convert confidence value to a pair of floats [forward, reverse]."""
      try:
        if isinstance(value, list):
          if len(value) >= 2:
            return [float(value[0]), float(value[1])]
          elif len(value) == 1:
            return [float(value[0]), 0.0]
          else:
            return [0.0, 0.0]
        elif isinstance(value, (int, float)):
          return [float(value), 0.0]
        else:
          return [0.0, 0.0]
      except Exception:
        return [0.0, 0.0]

    def _normalize_relation_type(self, rel: str) -> str:
      """Normalize relation types to standard categories for consistency."""
      if not isinstance(rel, str):
        return "UNKNOWN"
      r = rel.strip().lower()
      mapping = {
        "positive_regulate":"POSITIVE_REGULATE",
        "activate":"POSITIVE_REGULATE",
        "activates":"POSITIVE_REGULATE",
        "increase":"POSITIVE_REGULATE",
        "increases":"POSITIVE_REGULATE",
        "upregulate":"POSITIVE_REGULATE",
        "upregulates":"POSITIVE_REGULATE",

        "negative_regulate":"NEGATIVE_REGULATE",
        "inhibit":"NEGATIVE_REGULATE",
        "inhibits":"NEGATIVE_REGULATE",
        "decrease":"NEGATIVE_REGULATE",
        "decreases":"NEGATIVE_REGULATE",
        "downregulate":"NEGATIVE_REGULATE",
        "downregulates":"NEGATIVE_REGULATE",

        "causes":"CAUSES",
        "cause":"CAUSES",
        "results in":"CAUSES",
        "leads to":"CAUSES",
        "trigger":"CAUSES",
        "triggers":"CAUSES",

        "treats":"TREATS",
        "treat":"TREATS",
        "cures":"TREATS",
        "alleviates":"TREATS",
        "prevents":"TREATS",

        "interacts":"INTERACTS",
        "interact":"INTERACTS",
        "binds":"INTERACTS",
        "bind":"INTERACTS",
        "complexes with":"INTERACTS",

        "associated":"ASSOCIATED",
        "associate":"ASSOCIATED",
        "associated with":"ASSOCIATED",
        "correlated":"ASSOCIATED",
        "correlated with":"ASSOCIATED",
        "linked":"ASSOCIATED",
        "linked to":"ASSOCIATED",
      }
      if r in mapping:
        return mapping[r]
      if r.upper() in {"POSITIVE_REGULATE","NEGATIVE_REGULATE","CAUSES","TREATS","INTERACTS","ASSOCIATED"}:
        return r.upper()
      return r.upper() if r else "UNKNOWN"
