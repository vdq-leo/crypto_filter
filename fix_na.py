with open("api/routers/predictive.py", "r") as f:
    content = f.read()
import re
content = re.sub(
    r"print\(temp_df.isna\(\).sum\(\)\); temp_df = temp_df.dropna\(\)\n\n\s+if temp_df.empty:\n\s+raise HTTPException\(status_code=400, detail=f\".*?\"\)",
    """nas = temp_df.isna().sum().to_dict()
        temp_df = temp_df.dropna()
        if temp_df.empty:
            raise HTTPException(status_code=400, detail=f"NAs={nas}")""",
    content
)
with open("api/routers/predictive.py", "w") as f:
    f.write(content)
