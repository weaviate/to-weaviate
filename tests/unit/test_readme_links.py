"""The links of README.md, which PyPI shows as the project description.

PyPI resolves a relative link against pypi.org, so every link is an absolute
URL or an anchor of the page.
"""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPOSITORY = Path(__file__).resolve().parents[2]
README = REPOSITORY / "README.md"

FENCED_BLOCK = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
# The target of [text](target), ![alt](target) and [text](target "title").
INLINE_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\)")
# A reference definition: [label]: target
REFERENCE = re.compile(r"^ {0,3}\[[^\]\n]+\]:\s*<?([^\s>]+)", re.MULTILINE)
HTML_ATTRIBUTE = re.compile(r"\b(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

# Both forms name a file of the default branch: /<owner>/<repo>/blob/main/<path>
# on github.com and /<owner>/<repo>/main/<path> on raw.githubusercontent.com.
FILE_ON_MAIN = {
    "github.com": "/weaviate/to-weaviate/blob/main/",
    "raw.githubusercontent.com": "/weaviate/to-weaviate/main/",
}


def link_targets(markdown: str) -> list[str]:
    # Code is removed first: the text inside it is not a link. The link text
    # of [`code`](target) is removed with it, the target stays.
    prose = INLINE_CODE.sub("``", FENCED_BLOCK.sub("", markdown))
    return [
        target
        for pattern in (INLINE_LINK, REFERENCE, HTML_ATTRIBUTE)
        for target in pattern.findall(prose)
    ]


def readme_targets() -> list[str]:
    return link_targets(README.read_text(encoding="utf-8"))


def repository_path(target: str) -> str | None:
    url = urlsplit(target)
    prefix = FILE_ON_MAIN.get(url.netloc)
    if prefix is None or not url.path.startswith(prefix):
        return None
    return unquote(url.path[len(prefix) :])


def test_link_targets_finds_every_form_of_link() -> None:
    markdown = "\n".join(
        [
            "[![badge](https://img.example/b.svg)](LICENSE)",
            '![alt (with brackets)](docs/demo.gif "title")',
            "[`docs/seed.py`](docs/seed.py) and [anchor](#install)",
            "[label]: docs/reference.md",
            '<a href="docs/html.md"><img src="docs/html.png"></a>',
            "`[not a link](in-code.md)`",
            "```bash",
            "echo '[not a link](in-fence.md)'",
            "```",
        ]
    )

    assert sorted(link_targets(markdown)) == [
        "#install",
        "LICENSE",
        "docs/demo.gif",
        "docs/html.md",
        "docs/html.png",
        "docs/reference.md",
        "docs/seed.py",
        "https://img.example/b.svg",
    ]


def test_the_readme_has_links() -> None:
    assert len(readme_targets()) >= 20


def test_every_readme_link_is_absolute_or_an_anchor() -> None:
    relative = [
        target
        for target in readme_targets()
        if not target.startswith(("https://", "#"))
    ]

    assert relative == [], (
        "These README.md links are neither https:// URLs nor #anchors. PyPI "
        "shows README.md and resolves a relative link against pypi.org. Write "
        "a link to a file of the repository as "
        "https://github.com/weaviate/to-weaviate/blob/main/<path>. Write any "
        "other link as an https:// URL."
    )


def test_every_readme_link_to_a_file_on_main_names_a_file_of_the_repository() -> None:
    paths = [path for target in readme_targets() if (path := repository_path(target))]
    missing = [path for path in paths if not (REPOSITORY / path).is_file()]

    assert paths, "README.md links to no file of the repository"
    assert missing == []
