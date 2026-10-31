#Day 1: Literature Search Agent
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class AgentState:
    question: str
    search_queries: List[str] = field(default_factory=list)
    papers: List[Dict] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    iteration: int = 0
    status: str = "running"
    errors: List[str] = field(default_factory=list)
state = AgentState(
    question="What mechanisms contribute to AMR in E. coli?"
)

print("Question:", state.question)
print("Iteration:", state.iteration)
print("Status:", state.status)
print("Papers:", state.papers)
state.search_queries.append("E. coli antibiotic resistance mechanisms")

state.gaps.append("Limited evidence on regional trends in E. coli antimicrobial resistance.")

state.iteration = 1

print("Search queries:", state.search_queries)
print("Research gaps:", state.gaps)
print("Iteration:", state.iteration)
print("Day 2: <------------------------------------------------------------>")
#Day 2: Literature Search Agent

import requests
import time

def search_europe_pmc(query: str, max_results: int = 5) -> list:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    params = {
        "query": query,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core"
    }

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < 2:
                    print(f"Server error {response.status_code}. Retrying...")
                    time.sleep(2 ** (attempt + 1))
                    continue

            response.raise_for_status()

            results = response.json().get(
                "resultList", {}
            ).get("result", [])

            papers = []

            for paper in results:
                papers.append({
                    "title": paper.get("title", "Title unavailable"),
                    "year": paper.get(
                        "firstPublicationDate", "Year unavailable"
                    ),
                    "pmid": paper.get("pmid"),
                    "doi": paper.get("doi"),
                    "abstract": paper.get("abstractText", ""),
                    "source": "Europe PMC"
                })

            return papers

        except requests.RequestException as error:
            if attempt == 2:
                raise
            print(f"Request failed: {error}. Retrying...")
            time.sleep(2 ** (attempt + 1))

    return []

papers = search_europe_pmc(
    "Escherichia coli antimicrobial resistance mechanisms",
    max_results=5
)

for paper in papers:
    print(paper["title"])
    print("PMID:", paper["pmid"], "| DOI:", paper["doi"])
    print("-" * 60)
def paper_url(paper: dict) -> str:
    if paper.get("doi"):
        return "https://doi.org/" + paper["doi"]

    if paper.get("pmid"):
        return "https://pubmed.ncbi.nlm.nih.gov/" + str(paper["pmid"]) + "/"

    return "https://europepmc.org/search?query=" + requests.utils.quote(
        paper.get("title", "")
    )
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class LiteratureInput(BaseModel):
    query: str = Field(description="Focused scientific literature query")
    max_results: int = Field(default=5, ge=1, le=10)

@tool(args_schema=LiteratureInput)
def literature_search_tool(query: str, max_results: int = 5) -> str:
    """Search Europe PMC for biomedical research papers.
    Use for antimicrobial resistance mechanisms, bacterial species,
    resistance genes, and scientific evidence. Returns paper titles,
    publication dates, identifiers, abstracts, and source information."""
    try:
        results = search_europe_pmc(query, max_results)

        if not results:
            return f"No papers found for query: {query}"

        output = []

        for paper in results:
            output.append(
                f"Title: {paper['title']}\n"
                f"Year: {paper['year']}\n"
                f"PMID: {paper['pmid']}\n"
                f"DOI: {paper['doi']}\n"
                f"Abstract: {paper['abstract'][:1500]}\n"
                f"URL: {paper_url(paper)}"
            )

        return "\n\n".join(output)

    except requests.RequestException as error:
        return f"Literature search error: {error}"
result = literature_search_tool.invoke({
    "query": "E. coli beta-lactam resistance",
    "max_results": 3
})

print(result)
