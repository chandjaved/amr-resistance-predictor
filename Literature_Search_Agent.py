
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote
import os
import time

import requests


EUROPE_PMC_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class AgentState:
    question: str
    search_queries: List[str] = field(default_factory=list)
    papers: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    iteration: int = 0
    status: str = "running"
    errors: List[str] = field(default_factory=list)


MECHANISM_KEYWORDS = {
    "Beta-lactamases": [
        "beta-lactamase",
        "esbl",
        "bla",
        "carbapenemase",
    ],
    "Efflux pumps": [
        "efflux pump",
        "efflux system",
    ],
    "Reduced permeability": [
        "porin",
        "outer membrane",
        "reduced permeability",
    ],
    "Target modification": [
        "target modification",
        "altered target",
        "penicillin-binding protein",
    ],
    "Horizontal gene transfer": [
        "horizontal gene transfer",
        "conjugation",
        "plasmid transfer",
    ],
    "Biofilm-associated resistance": [
        "biofilm",
        "biofilm-associated resistance",
    ],
    "Plasmid-mediated resistance": [
        "plasmid-mediated resistance",
        "resistance plasmid",
    ],
    "Integrons and mobile genetic elements": [
        "integron",
        "transposon",
        "mobile genetic element",
    ],
}


def search_europe_pmc(
    query: str,
    max_results: int = 5,
    max_retries: int = 4,
) -> List[Dict[str, Any]]:

    if not query or not query.strip():
        raise ValueError("The literature search query cannot be empty.")

    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100.")

    params = {
        "query": query.strip(),
        "format": "json",
        "pageSize": max_results,
        "resultType": "core",
    }

    headers = {
        "User-Agent": "AMR-Literature-Search-Agent/1.0",
        "Accept": "application/json",
    }

    last_error = None

    for attempt in range(max_retries):

        try:
            response = requests.get(
                EUROPE_PMC_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

            if response.status_code in RETRYABLE_STATUS_CODES:

                last_error = (
                    f"Europe PMC returned HTTP "
                    f"{response.status_code}"
                )

                if attempt < max_retries - 1:

                    retry_after = response.headers.get("Retry-After")

                    try:
                        wait_seconds = min(
                            float(retry_after), 60
                        )
                    except (TypeError, ValueError):
                        wait_seconds = 2 ** (attempt + 1)

                    time.sleep(wait_seconds)
                    continue

            response.raise_for_status()

            payload = response.json()

            raw_results = (
                payload.get("resultList", {}).get("result", [])
            )

            papers = []

            for item in raw_results:

                abstract = item.get("abstractText") or ""

                papers.append({
                    "title": item.get("title") or "Title unavailable",
                    "year": (
                        item.get("firstPublicationDate")
                        or item.get("pubYear")
                        or "Year unavailable"
                    ),
                    "pmid": item.get("pmid") or "",
                    "doi": item.get("doi") or "",
                    "abstract": abstract,
                    "abstractText": abstract,
                    "authorString": (
                        item.get("authorString") or "Not available"
                    ),
                    "journalTitle": (
                        item.get("journalTitle") or "Not available"
                    ),
                    "source": "Europe PMC",
                })

            return papers

        except requests.HTTPError as error:

            last_error = str(error)

            status_code = (
                error.response.status_code
                if error.response is not None
                else None
            )

            if (
                status_code not in RETRYABLE_STATUS_CODES
                or attempt == max_retries - 1
            ):
                break

            time.sleep(2 ** (attempt + 1))

        except requests.RequestException as error:

            last_error = str(error)

            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

        except ValueError as error:

            last_error = f"Invalid Europe PMC JSON response: {error}"

            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        f"Europe PMC search failed after {max_retries} attempts: "
        f"{last_error}"
    )


def paper_url(paper: Dict[str, Any]) -> str:

    doi = paper.get("doi")
    pmid = paper.get("pmid")

    if doi:
        return f"https://doi.org/{doi}"

    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    title = paper.get("title", "")

    return (
        "https://europepmc.org/search?query="
        + quote(str(title))
    )


def deduplicate_papers(
    papers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    unique = {}

    for paper in papers:

        doi = str(paper.get("doi") or "").strip().lower()
        pmid = str(paper.get("pmid") or "").strip()
        title = str(paper.get("title") or "").strip().lower()

        if doi:
            key = "doi:" + doi
        elif pmid:
            key = "pmid:" + pmid
        elif title:
            key = "title:" + title
        else:
            continue

        if key not in unique:
            unique[key] = paper

    return list(unique.values())


def identify_candidate_gaps(
    question: str,
    papers: List[Dict[str, Any]]
) -> List[str]:

    combined_text = " ".join(
        (
            str(paper.get("title", ""))
            + " "
            + str(
                paper.get("abstract")
                or paper.get("abstractText")
                or ""
            )
        ).lower()
        for paper in papers
    )

    gaps = []

    for mechanism, keywords in MECHANISM_KEYWORDS.items():

        if not any(
            keyword.lower() in combined_text
            for keyword in keywords
        ):
            gaps.append(mechanism)

    return gaps


def make_followup_query(
    question: str,
    gap: str
) -> str:

    return f'({question.strip()}) AND ("{gap}")'


def choose_next_action(
    question: str,
    papers: List[Dict[str, Any]],
    gaps: List[str],
    searched_gaps: Optional[List[str]] = None,
) -> Dict[str, Any]:

    searched = set(searched_gaps or [])

    for gap in gaps:

        if gap not in searched:

            return {
                "action": "search",
                "gap": gap,
                "query": make_followup_query(question, gap),
            }

    return {
        "action": "finish",
        "query": None,
        "reason": "No unsearched candidate gaps remain.",
    }


def should_stop(
    iteration: int,
    max_iterations: int,
    paper_count: int,
    remaining_gaps: List[str],
    repeated_query: bool = False,
) -> bool:

    return (
        iteration >= max_iterations
        or paper_count == 0
        or not remaining_gaps
        or repeated_query
    )


def run_literature_research(
    question: str,
    max_iterations: int = 3,
    results_per_search: int = 5,
) -> Dict[str, Any]:

    if not question or not question.strip():
        raise ValueError("Please provide a research question.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    if not 1 <= results_per_search <= 100:
        raise ValueError(
            "results_per_search must be between 1 and 100."
        )

    question = question.strip()

    max_iterations = min(int(max_iterations), 5)

    state = AgentState(question=question)

    searched_queries = set()
    searched_gaps = set()

    for iteration in range(max_iterations):

        state.iteration = iteration + 1

        if iteration == 0:

            action = {
                "action": "search",
                "gap": "initial research question",
                "query": question,
            }

        else:

            action = choose_next_action(
                question=question,
                papers=state.papers,
                gaps=state.gaps,
                searched_gaps=list(searched_gaps),
            )

        if action["action"] == "finish":
            state.status = "complete"
            break

        query = action["query"].strip()
        gap = action.get("gap", "")

        query_key = query.lower()

        if query_key in searched_queries:
            state.status = "stopped_repeated_query"
            break

        searched_queries.add(query_key)
        state.search_queries.append(query)

        if gap and gap != "initial research question":
            searched_gaps.add(gap)

        try:

            new_papers = search_europe_pmc(
                query=query,
                max_results=results_per_search,
            )

        except (RuntimeError, ValueError) as error:

            state.errors.append(
                f"{query}: {error}"
            )

            state.status = "stopped_with_errors"
            break

        if not new_papers:

            state.errors.append(
                f"No papers returned for query: {query}"
            )

            state.status = (
                "no_results"
                if not state.papers
                else "completed_with_warnings"
            )

            break

        state.papers = deduplicate_papers(
            state.papers + new_papers
        )

        state.gaps = identify_candidate_gaps(
            question,
            state.papers,
        )

        remaining_gaps = [
            candidate_gap
            for candidate_gap in state.gaps
            if candidate_gap not in searched_gaps
        ]

        if should_stop(
            iteration=state.iteration,
            max_iterations=max_iterations,
            paper_count=len(state.papers),
            remaining_gaps=remaining_gaps,
        ):

            if not remaining_gaps:
                state.status = "complete"

            break

    if state.status == "running":

        state.status = (
            "stopped_with_errors"
            if state.errors
            else "max_iterations_reached"
        )

    return {
        "question": state.question,
        "search_queries": state.search_queries,
        "papers": state.papers,
        "gaps": state.gaps,
        "iteration": state.iteration,
        "status": state.status,
        "errors": state.errors,

        # Backward-compatible result keys
        "queries": state.search_queries,
        "candidate_gaps": state.gaps,
        "iterations": state.iteration,
    }


def prepare_evidence(
    papers: List[Dict[str, Any]],
    max_abstract_chars: int = 1200,
) -> List[Dict[str, Any]]:

    evidence = []

    for paper in papers:

        abstract = (
            paper.get("abstract")
            or paper.get("abstractText")
            or "Abstract unavailable"
        )

        evidence.append({
            "title": paper.get("title", "Title unavailable"),
            "year": paper.get("year", "Year unavailable"),
            "pmid": paper.get("pmid") or "Not available",
            "doi": paper.get("doi") or "Not available",
            "url": paper_url(paper),
            "abstract": abstract[:max_abstract_chars],
        })

    return evidence


def format_references(
    papers: List[Dict[str, Any]]
) -> str:

    lines = []

    for index, paper in enumerate(papers, start=1):

        lines.append(
            f"[{index}] "
            f"{paper.get('title', 'Title unavailable')}. "
            f"{paper.get('year', 'Year unavailable')}. "
            f"{paper_url(paper)}"
        )

    return "\n".join(lines)


USE_GEMINI = False
GEMINI_MODEL = "gemini-2.5-flash"


def initialize_gemini():

    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )

    if not api_key:
        return None

    try:

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.2,
        )

    except Exception:
        return None


def synthesize_literature(
    question: str,
    papers: List[Dict[str, Any]],
    llm: Optional[Any],
) -> str:

    if not papers:
        return "No retrieved papers are available to synthesize."

    if llm is None:

        return (
            "AI synthesis was skipped. Retrieved papers and "
            "references are still available."
        )

    evidence = prepare_evidence(papers[:10])

    evidence_text = "\n\n".join(
        (
            f"Paper [{index}]\n"
            f"Title: {item['title']}\n"
            f"Year: {item['year']}\n"
            f"PMID: {item['pmid']}\n"
            f"DOI: {item['doi']}\n"
            f"URL: {item['url']}\n"
            f"Abstract: {item['abstract']}"
        )
        for index, item in enumerate(evidence, start=1)
    )

    prompt = f"""
You are an academic biomedical literature research assistant.

RESEARCH QUESTION:
{question}

RETRIEVED EUROPE PMC EVIDENCE:
{evidence_text}

Instructions:
1. Summarize mechanisms discussed in the supplied papers.
2. Support claims with numbered citations.
3. Use only information supported by the supplied evidence.
4. Identify evidence gaps and limitations.
5. Distinguish established findings from uncertain findings.
6. Do not invent authors, results, or citations.
7. Do not claim the retrieved papers represent all available literature.
8. State when abstracts provide insufficient evidence.

Use these headings:
1. Brief Literature Summary
2. Mechanisms and Supporting Evidence
3. Evidence Gaps and Limitations
4. Overall Conclusion
5. References with URLs
"""

    try:

        response = llm.invoke(prompt)

        return str(response.content)

    except Exception as error:

        return f"Gemini synthesis failed: {error}"


def main() -> None:

    question = (
        "What mechanisms contribute to antimicrobial "
        "resistance in Escherichia coli?"
    )

    print("\n" + "=" * 65)
    print("STAGE 3 — LITERATURE SEARCH AGENT")
    print("=" * 65)

    print("Research question:", question)

    report = run_literature_research(
        question=question,
        max_iterations=3,
        results_per_search=5,
    )

    print("\nRESEARCH RESULTS")
    print("-" * 65)

    print("Status:", report["status"])
    print("Iterations:", report["iterations"])
    print("Queries:", report["queries"])
    print("Unique papers:", len(report["papers"]))
    print("Candidate coverage gaps:", report["candidate_gaps"])

    if report["errors"]:

        print("\nErrors / warnings:")

        for error in report["errors"]:
            print("-", error)

    print("\nRETRIEVED PAPERS")
    print("-" * 65)

    for index, paper in enumerate(report["papers"], start=1):

        print(f"\n{index}. {paper['title']}")
        print("Year:", paper["year"])
        print("PMID:", paper["pmid"])
        print("DOI:", paper["doi"])
        print("URL:", paper_url(paper))

    print("\nREFERENCES")
    print("-" * 65)

    print(
        format_references(report["papers"][:10])
        or "No references available."
    )

    if USE_GEMINI:

        print("\nGEMINI LITERATURE SYNTHESIS")
        print("-" * 65)

        llm = initialize_gemini()

        summary = synthesize_literature(
            question=report["question"],
            papers=report["papers"],
            llm=llm,
        )

        print(summary)

    else:

        print(
            "\nGemini is disabled (USE_GEMINI = False). "
            "No LLM API call or LLM tokens were used."
        )


if __name__ == "__main__":
    main()
