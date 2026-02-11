import httpx
from config import SEARXNG_BASE_URL


async def search_web(query: str, max_results: int = 5) -> str:
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{SEARXNG_BASE_URL}/search",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for i, item in enumerate(data.get("results", [])[:max_results], 1):
                title = item.get("title", "Untitled")
                url = item.get("url", "")
                snippet = item.get("content", "")
                results.append(f"[{i}] {title}\nURL: {url}\n{snippet}")

            return "\n\n".join(results) if results else "No search results found."

        except httpx.HTTPError:
            return "Search unavailable."


def read_file(file_path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """
    Read the content of a file. Supports line ranges.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total_lines = len(lines)

            if start_line is None:
                start_line = 1
            if end_line is None:
                end_line = total_lines

            # Validate range
            start_line = max(1, start_line)
            end_line = min(total_lines, end_line)

            if start_line > end_line:
                return f"Error: start_line ({start_line}) cannot be greater than end_line ({end_line})."

            selected_lines = lines[start_line - 1 : end_line]
            content = "".join(selected_lines)
            
            # Add line numbers for context if reading a subset
            numbered_content = []
            for i, line in enumerate(selected_lines, start_line):
                numbered_content.append(f"{i}: {line}")
            
            return "".join(numbered_content)

    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
