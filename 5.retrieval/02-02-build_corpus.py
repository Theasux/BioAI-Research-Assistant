import json
import time
import requests



# 1. PubMed API 地址


SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"



# 2. 设置多个 PubMed 检索词


SEARCH_TERMS = [
    "maize salt stress",
    "maize salt tolerance",
    "maize salt stress transcriptome",
    "maize alternative splicing salt stress",
    "plant salt stress alternative splicing",
]


# 每个检索词最多获取多少篇论文
RETMAX_PER_QUERY = 20



# 3. PubMed 搜索函数


def search_pubmed(term, retmax=20):
    """
    根据关键词搜索 PubMed，返回 PMID 列表。
    """

    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
    }

    response = requests.get(
        SEARCH_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    pmids = data["esearchresult"]["idlist"]

    return pmids



# 4. 获取论文详细信息


def fetch_pubmed_articles(pmids):
    """
    根据 PMID 列表获取论文 XML，并提取：
    PMID
    Title
    Abstract
    Authors
    """

    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = requests.get(
        FETCH_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    root = __import__("xml.etree.ElementTree", fromlist=[""]).fromstring(
        response.text
    )

    articles = []

    for article in root.findall(".//PubmedArticle"):

        # ----------------------------------------------------
        # PMID
        # ----------------------------------------------------

        pmid_element = article.find(".//PMID")

        if pmid_element is None:
            continue

        pmid = pmid_element.text


        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title_element = article.find(".//ArticleTitle")

        if title_element is None:
            title = ""

        else:
            title = "".join(title_element.itertext()).strip()


        # ----------------------------------------------------
        # Abstract
        # ----------------------------------------------------

        abstract_elements = article.findall(".//AbstractText")

        abstract_parts = []

        for abstract_element in abstract_elements:

            text = "".join(abstract_element.itertext()).strip()

            if text:

                label = abstract_element.attrib.get("Label")

                if label:
                    text = f"{label}: {text}"

                abstract_parts.append(text)

        abstract = " ".join(abstract_parts)


        # ----------------------------------------------------
        # Authors
        # ----------------------------------------------------

        authors = []

        for author in article.findall(".//Author"):

            lastname = author.find("LastName")
            firstname = author.find("ForeName")

            if lastname is not None and firstname is not None:

                authors.append(
                    f"{firstname.text} {lastname.text}"
                )


        # ----------------------------------------------------
        # 保存论文
        # ----------------------------------------------------

        if abstract:

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                }
            )

    return articles



# 5. 主程序


def main():

    print("=" * 70)
    print("Building PubMed Literature Corpus")
    print("=" * 70)


    # --------------------------------------------------------
    # 第一步：执行多个 PubMed 搜索
    # --------------------------------------------------------

    all_pmids = []

    for term in SEARCH_TERMS:

        print()
        print(f"Searching PubMed: {term}")

        pmids = search_pubmed(
            term,
            RETMAX_PER_QUERY
        )

        print(
            f"Found {len(pmids)} papers"
        )

        all_pmids.extend(pmids)

        # 稍微等待，避免连续请求过快
        time.sleep(0.5)


    # --------------------------------------------------------
    # 第二步：PMID 去重
    # --------------------------------------------------------

    unique_pmids = list(
        dict.fromkeys(all_pmids)
    )

    print()
    print("=" * 70)
    print("PMID Deduplication")
    print("=" * 70)

    print(
        f"Total PMID before deduplication: {len(all_pmids)}"
    )

    print(
        f"Unique PMID after deduplication: {len(unique_pmids)}"
    )


    # --------------------------------------------------------
    # 第三步：获取论文详细内容
    # --------------------------------------------------------

    print()
    print("Fetching article contents...")

    all_articles = []

    # PubMed 一次请求一批 PMID
    batch_size = 100

    for i in range(
        0,
        len(unique_pmids),
        batch_size
    ):

        batch_pmids = unique_pmids[
            i:i + batch_size
        ]

        print(
            f"Fetching batch "
            f"{i + 1} - "
            f"{i + len(batch_pmids)}"
        )

        articles = fetch_pubmed_articles(
            batch_pmids
        )

        all_articles.extend(articles)

        time.sleep(0.5)


    # --------------------------------------------------------
    # 第四步：再次按照 PMID 去重
    # --------------------------------------------------------

    article_dict = {}

    for article in all_articles:

        pmid = article["pmid"]

        article_dict[pmid] = article


    articles = list(
        article_dict.values()
    )


    # --------------------------------------------------------
    # 第五步：保存 Corpus
    # --------------------------------------------------------

    output_file = (
        "5.retrieval/pubmed_corpus.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            articles,
            f,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------------
    # 第六步：输出统计信息
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Corpus Construction Finished")
    print("=" * 70)

    print(
        f"Articles with abstracts: {len(articles)}"
    )

    print(
        f"Saved to: {output_file}"
    )


    # --------------------------------------------------------
    # 打印前5篇论文
    # --------------------------------------------------------

    print()
    print("First 5 papers:")

    for i, article in enumerate(
        articles[:5],
        start=1
    ):

        print()
        print(f"[{i}] PMID: {article['pmid']}")
        print(f"Title: {article['title']}")



# 7. 程序入口


if __name__ == "__main__":
    main()