import json
import time
from pathlib import Path
import requests
import xml.etree.ElementTree as ET



# 1. 基本配置


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "pubmed_corpus_v2.json"

EMAIL = "1182087541@qq.com"

RETMAX_PER_QUERY = 40

SEARCH_GROUPS = {
    "salt_stress": [
        "maize salt stress",
        "maize salt tolerance",
        "plant salt stress",
        "plant salinity stress",
    ],

    "alternative_splicing": [
        "maize alternative splicing salt stress",
        "plant alternative splicing salt stress",
        "alternative splicing salinity plants",
        "alternative splicing salt tolerance plants",
    ],

    "ion_transport": [
        "plant sodium uptake salt stress",
        "plant sodium transport salt stress",
        "plant Na+ transport salt stress",
        "HKT sodium transport plant",
        "SOS1 sodium transport plant",
        "NHX salt stress plant",
        "potassium sodium homeostasis plant salt stress",
    ],

    "photosynthesis": [
        "maize photosynthesis salt stress",
        "plant photosynthesis salinity",
        "plant ROS salt stress",
        "plant oxidative stress salt stress",
        "plant ion homeostasis salt stress",
    ],

    "maize_regulation": [
        "maize salt stress transcriptome",
        "maize salt tolerance genes",
        "maize salt stress gene regulation",
        "maize ion homeostasis salt stress",
        "maize salt stress molecular mechanism",
    ],
}



# 2. PubMed API


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def pubmed_search(term, retmax=40):
    """
    根据检索词从 PubMed 获取 PMID。
    """

    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "email": EMAIL,
    }

    response = requests.get(
        ESEARCH_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data["esearchresult"]["idlist"]


def pubmed_fetch(pmids):
    """
    根据 PMID 批量获取 PubMed XML。
    """

    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": EMAIL,
    }

    response = requests.get(
        EFETCH_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return ET.fromstring(response.text)



# 3. XML 解析


def extract_article(article):
    """
    从一个 PubmedArticle XML 节点中提取结构化信息。
    """

    # -------------------------
    # PMID
    # -------------------------
    pmid_node = article.find(".//PMID")

    if pmid_node is None:
        return None

    pmid = pmid_node.text


    # -------------------------
    # Title
    # -------------------------
    title_node = article.find(".//ArticleTitle")

    if title_node is None:
        title = ""

    else:
        title = "".join(title_node.itertext()).strip()


    # -------------------------
    # Abstract
    # -------------------------
    abstract_parts = []

    for abstract_node in article.findall(".//AbstractText"):
        text = "".join(abstract_node.itertext()).strip()

        label = abstract_node.attrib.get("Label")

        if label:
            text = f"{label}: {text}"

        if text:
            abstract_parts.append(text)

    abstract = " ".join(abstract_parts)


    # -------------------------
    # Authors
    # -------------------------
    authors = []

    for author in article.findall(".//Author"):

        lastname = author.find("LastName")
        firstname = author.find("ForeName")

        if lastname is not None and firstname is not None:

            authors.append(
                f"{firstname.text} {lastname.text}"
            )


    # -------------------------
    # Journal
    # -------------------------
    journal_node = article.find(".//Journal/Title")

    if journal_node is not None:
        journal = journal_node.text or ""
    else:
        journal = ""


    # -------------------------
    # Publication year
    # -------------------------
    year = ""

    year_node = article.find(".//PubDate/Year")

    if year_node is not None:
        year = year_node.text or ""

    else:
        medline_date = article.find(".//PubDate/MedlineDate")

        if medline_date is not None:
            year = (medline_date.text or "")[:4]


    # -------------------------
    # DOI
    # -------------------------
    doi = ""

    for article_id in article.findall(".//ArticleId"):

        if article_id.attrib.get("IdType") == "doi":

            doi = article_id.text or ""

            break


    # -------------------------
    # 返回
    # -------------------------

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "year": year,
        "doi": doi,
    }



# 4. 第一步：检索 PMID


print("=" * 70)
print("BioAI PubMed Corpus V2 Builder")
print("=" * 70)

all_records = []

for topic, queries in SEARCH_GROUPS.items():

    print()
    print("=" * 70)
    print(f"TOPIC: {topic}")
    print("=" * 70)

    for query in queries:

        print(f"Searching: {query}")

        try:

            pmids = pubmed_search(
                query,
                RETMAX_PER_QUERY
            )

            print(f"  Found: {len(pmids)}")

            for pmid in pmids:

                all_records.append(
                    {
                        "pmid": pmid,
                        "topic": topic,
                        "query": query,
                    }
                )

            time.sleep(0.4)

        except Exception as e:

            print(f"  ERROR: {e}")


print()
print("=" * 70)
print("PMID collection summary")
print("=" * 70)

print(
    f"Total PMID-query records: {len(all_records)}"
)



# 5. PMID 去重


pmid_to_topics = {}

pmid_to_queries = {}

for record in all_records:

    pmid = record["pmid"]

    topic = record["topic"]

    query = record["query"]


    if pmid not in pmid_to_topics:

        pmid_to_topics[pmid] = set()

    pmid_to_topics[pmid].add(topic)


    if pmid not in pmid_to_queries:

        pmid_to_queries[pmid] = set()

    pmid_to_queries[pmid].add(query)


unique_pmids = list(pmid_to_topics.keys())


print(
    f"Unique PMIDs after deduplication: "
    f"{len(unique_pmids)}"
)



# 6. 批量获取论文


articles = []

BATCH_SIZE = 100

for start in range(
    0,
    len(unique_pmids),
    BATCH_SIZE
):

    batch = unique_pmids[
        start:start + BATCH_SIZE
    ]

    print()
    print(
        f"Fetching batch "
        f"{start + 1} - "
        f"{start + len(batch)}"
    )

    try:

        root = pubmed_fetch(batch)

        for article in root.findall(".//PubmedArticle"):

            record = extract_article(article)

            if record is None:
                continue


            pmid = record["pmid"]


            # 添加主题
            record["topics"] = sorted(
                pmid_to_topics.get(
                    pmid,
                    []
                )
            )


            # 添加来源 query
            record["queries"] = sorted(
                pmid_to_queries.get(
                    pmid,
                    []
                )
            )


            articles.append(record)


        time.sleep(0.5)

    except Exception as e:

        print(f"ERROR: {e}")



# 7. 去掉没有 Abstract 的论文


articles_with_abstract = [
    article
    for article in articles
    if article["abstract"].strip()
]


print()
print("=" * 70)
print("Final corpus")
print("=" * 70)

print(
    f"Articles fetched: "
    f"{len(articles)}"
)

print(
    f"Articles with abstracts: "
    f"{len(articles_with_abstract)}"
)



# 8. 按 PMID 排序


articles_with_abstract.sort(
    key=lambda x: int(x["pmid"])
    if x["pmid"].isdigit()
    else 0,
    reverse=True
)



# 9. 保存 JSON


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        articles_with_abstract,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 70)
print("Saved")
print("=" * 70)

print(OUTPUT_PATH)



# 10. 统计每个主题


topic_counts = {}

for article in articles_with_abstract:

    for topic in article["topics"]:

        topic_counts[topic] = (
            topic_counts.get(topic, 0) + 1
        )


print()
print("Topic distribution:")

for topic, count in sorted(
    topic_counts.items()
):

    print(
        f"{topic:25s} : {count}"
    )



# 11. 打印前 5 篇


print()
print("=" * 70)
print("First 5 papers")
print("=" * 70)

for article in articles_with_abstract[:5]:

    print()
    print(
        f"PMID: {article['pmid']}"
    )

    print(
        f"Title: {article['title']}"
    )

    print(
        f"Topics: "
        f"{', '.join(article['topics'])}"
    )