import pandas as pd

# --- STEP 1: Load CSVs ---
df_sri1 = pd.read_csv("cuisine_srilanka_b1.csv")
df_sri2 = pd.read_csv("cuisine_srilanka_b2.csv")
df_global = pd.read_csv("cuisine_globle.csv", sep='\t')  # Use tab if needed

# --- STEP 2: Combine Sri Lankan dishes ---
df_sri = pd.concat([df_sri1, df_sri2], ignore_index=True)
df_sri["cuisine_type"] = "sri_lankan"

# --- STEP 3: Process Global Cuisines into Dish Format ---
global_dishes = []
for _, row in df_global.iterrows():
    cuisines = str(row['cusine_type']).split(',')
    for cuisine in cuisines:
        global_dishes.append({
            "dish_name": cuisine.strip(),
            "description": None,
            "key_ingredients": None,
            "sub_category": None,
            "region_notes": row.get("City", None),
            "cuisine_type": "global"
        })
df_global_dishes = pd.DataFrame(global_dishes)

# --- STEP 4: Standardize Column Names ---
required_cols = ["dish_name", "description", "key_ingredients", "sub_category", "region_notes", "cuisine_type"]
df_sri = df_sri[["dish_name", "description", "key_ingredients", "sub_category", "region_notes", "cuisine_type"]]
df_global_dishes = df_global_dishes[required_cols]

# --- STEP 5: Combine All Data ---
df_all = pd.concat([df_sri, df_global_dishes], ignore_index=True)

# --- STEP 6: Define Cuisine Labels ---
cuisine_labels = [
    "Rice", "Rice and Curry", "Curry", "South Indian", "North Indian", "Kottu", "Street Food", "Short Eats",
    "Seafood", "Vegetarian", "Jaffna Tamil Cuisine", "Sri Lankan BBQ & Grill", "Chinese", "Thai", "Japanese",
    "Korean", "Italian", "Pasta", "Pizza", "Fast Food", "French", "Mediterranean", "Western Contemporary",
    "Arabian", "Cafe & Beverages"
]

# --- STEP 7: Optional Rule-Based Cuisine Labeling ---
def assign_label(row):
    name = str(row['dish_name']).lower()
    desc = str(row['description']).lower() if row['description'] else ""
    ingredients = str(row['key_ingredients']).lower() if row['key_ingredients'] else ""

    for label in cuisine_labels:
        l = label.lower()
        if l in name or l in desc or l in ingredients:
            return label
    return "Other"

df_all["cuisine_label"] = df_all.apply(assign_label, axis=1)

# --- STEP 8: Save Final ML-Ready CSV ---
df_all.to_csv("Foodcusine.csv", index=False)
