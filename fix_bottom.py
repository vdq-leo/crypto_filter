import os
import re

directory = "modules"

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r") as f:
                content = f.read()
            
            # Replace cases with params already present
            new_content = re.sub(
                r'requests\.get\(f"\{API_BASE_URL\}/data/universe",\s*params=\{([^}]+)\}\)',
                lambda m: f'requests.get(f"{{API_BASE_URL}}/data/universe", params={{{m.group(1)}, "bottom": str(input.quick_vol_bottom()).lower()}})',
                content
            )
            
            # Replace cases without params
            new_content = re.sub(
                r'requests\.get\(f"\{API_BASE_URL\}/data/universe"\)',
                r'requests.get(f"{API_BASE_URL}/data/universe", params={"bottom": str(input.quick_vol_bottom()).lower()})',
                new_content
            )
            
            if new_content != content:
                with open(filepath, "w") as f:
                    f.write(new_content)
                print(f"Fixed {filepath}")

print("Done")
