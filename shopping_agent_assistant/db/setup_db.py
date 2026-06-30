import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlite_vec import serialize_float32

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "store.db"

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_product_text(name: str, category: str, description: str) -> str:
    return f"{name}. Category: {category}. Description: {description}"

def create_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            description TEXT,
            is_organic INTEGER DEFAULT 0
        )
    """)

    # Full-text index for keyword search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
            name,
            category,
            description,
            content='products',
            content_rowid='id'
        )
    """)

    # Vector storage for semantic search
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS product_embeddings (
            product_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            rating REAL,
            reviewer_name TEXT,
            review_text TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ordered_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    products = [
        # --- Honey (8) ---
        (1,  "Organic Raw Honey",             "honey",        14.99, "Pure organic raw honey, unfiltered and cold-pressed",                1),
        (2,  "Wildflower Honey",              "honey",        12.99, "Natural wildflower honey from local beekeepers",                     0),
        (3,  "Organic Manuka Honey",          "honey",        29.99, "Premium organic Manuka honey from New Zealand",                      1),
        (4,  "Clover Honey",                  "honey",         8.99, "Classic clover honey, smooth and sweet",                             0),
        (5,  "Organic Buckwheat Honey",       "honey",        18.99, "Dark and robust organic buckwheat honey, antioxidant-rich",          1),
        (6,  "Orange Blossom Honey",          "honey",        15.99, "Light and floral orange blossom honey",                              0),
        (7,  "Organic Acacia Honey",          "honey",        17.99, "Light and mild organic acacia honey, low glycemic index",            1),
        (8,  "Creamed Honey",                 "honey",        11.99, "Smooth creamed honey with spreadable texture",                       0),
        # --- Oils (4) ---
        (9,  "Organic Extra Virgin Olive Oil", "oil",         16.99, "Cold-pressed organic EVOO from Mediterranean olives",                1),
        (10, "Coconut Oil",                   "oil",          12.49, "Refined coconut oil, great for high-heat cooking",                   0),
        (11, "Organic Flaxseed Oil",          "oil",          14.99, "Cold-pressed organic flaxseed oil, rich in omega-3",                 1),
        (12, "Avocado Oil",                   "oil",          18.99, "Cold-pressed avocado oil, high smoke point",                         0),
        # --- Nuts & Seeds (4) ---
        (13, "Organic Almonds",               "nuts",         11.99, "Raw organic almonds, unsalted, non-GMO certified",                   1),
        (14, "Roasted Cashews",               "nuts",          9.99, "Lightly salted dry-roasted cashews",                                 0),
        (15, "Organic Chia Seeds",            "seeds",         8.49, "Organic black chia seeds, high in fiber and omega-3",                1),
        (16, "Mixed Nuts",                    "nuts",         13.99, "Premium mix of walnuts, pecans, almonds and Brazil nuts",            0),
        # --- Grains & Cereals (4) ---
        (17, "Organic Quinoa",                "grains",       10.99, "Organic white quinoa, complete protein, gluten-free",                1),
        (18, "Rolled Oats",                   "grains",        5.49, "Whole grain rolled oats, great for porridge and baking",             0),
        (19, "Organic Brown Rice",            "grains",        7.99, "Long-grain organic brown rice, naturally gluten-free",               1),
        (20, "Steel-Cut Oats",                "grains",        6.99, "Traditional steel-cut oats, low GI, hearty texture",                 0),
        # --- Tea & Coffee (4) ---
        (21, "Organic Green Tea",             "tea",          12.99, "Japanese organic sencha green tea, 50 bags",                         1),
        (22, "Chamomile Tea",                 "tea",           8.99, "Dried chamomile flowers, caffeine-free, soothing",                   0),
        (23, "Organic Ethiopian Coffee",      "coffee",       16.99, "Single-origin organic Arabica, medium roast whole bean",             1),
        (24, "Dark Roast Espresso Blend",     "coffee",       14.49, "Bold dark roast espresso blend, ground",                             0),
        # --- Snacks (4) ---
        (25, "Organic Granola",               "snacks",        9.99, "Organic oat granola with honey, almonds and dried cranberries",      1),
        (26, "Rice Cakes",                    "snacks",        4.49, "Lightly salted brown rice cakes, low calorie",                       0),
        (27, "Organic Dried Mango",           "snacks",        7.99, "Unsweetened organic dried mango slices, no preservatives",           1),
        (28, "Trail Mix",                     "snacks",        8.49, "Classic trail mix with raisins, M&Ms, peanuts and sunflower seeds",  0),
        # --- Dairy Alternatives (4) ---
        (29, "Organic Almond Milk",           "dairy-alt",     4.99, "Unsweetened organic almond milk, fortified with calcium",            1),
        (30, "Oat Milk",                      "dairy-alt",     4.49, "Barista-style oat milk, great for coffee",                           0),
        (31, "Organic Coconut Milk",          "dairy-alt",     3.99, "Full-fat organic coconut milk, great for curries",                   1),
        (32, "Soy Milk",                      "dairy-alt",     3.49, "Unsweetened soy milk, high protein",                                 0),
        # --- Superfoods (8) ---
        (33, "Organic Spirulina Powder",      "superfoods",    19.99, "Nutrient-dense blue-green algae powder for smoothies",               1),
        (34, "Cacao Nibs",                    "superfoods",    11.49, "Crunchy cacao nibs with a rich chocolate flavor",                     0),
        (35, "Organic Maca Powder",           "superfoods",    14.99, "Earthy maca root powder for energy-focused drinks",                   1),
        (36, "Hemp Hearts",                   "superfoods",    12.99, "Soft shelled hemp seeds with a mild nutty taste",                     0),
        (37, "Goji Berries",                  "superfoods",    10.99, "Dried goji berries, sweet and tangy superfruit snack",                0),
        (38, "Organic Matcha Powder",         "superfoods",    18.49, "Ceremonial-grade organic matcha for lattes and baking",               1),
        (39, "Pumpkin Seeds",                 "superfoods",     9.49, "Roasted pumpkin seeds packed with crunch and minerals",               0),
        (40, "Acai Powder",                   "superfoods",    17.99, "Freeze-dried acai powder for bowls and smoothies",                   0),
        # --- Pasta (8) ---
        (41, "Organic Spaghetti",             "pasta",          3.99, "Durum wheat spaghetti made with organic semolina",                    1),
        (42, "Penne Rigate",                  "pasta",          2.99, "Classic ridged penne pasta for hearty sauces",                        0),
        (43, "Chickpea Pasta",                "pasta",          4.99, "Protein-rich chickpea pasta, gluten-free",                            1),
        (44, "Brown Rice Pasta",              "pasta",          4.49, "Tender brown rice pasta with a mild flavor",                          0),
        (45, "Organic Fusilli",               "pasta",          3.79, "Organic spiral pasta that holds sauce beautifully",                  1),
        (46, "Lasagna Sheets",                "pasta",          4.29, "Flat lasagna sheets ready for layered bakes",                         0),
        (47, "Gluten-Free Lentil Pasta",      "pasta",          5.49, "Red lentil pasta with a hearty, satisfying texture",                  1),
        (48, "Whole Wheat Macaroni",          "pasta",          3.49, "Whole wheat macaroni for everyday comfort meals",                     0),
        # --- Sauces (8) ---
        (49, "Organic Marinara Sauce",        "sauces",         5.99, "Slow-simmered organic tomato sauce with basil and garlic",            1),
        (50, "Alfredo Sauce",                 "sauces",         6.49, "Creamy parmesan-style sauce for pasta and vegetables",                0),
        (51, "Organic Peanut Sauce",          "sauces",         4.99, "Savory organic peanut sauce for noodles and stir-fries",              1),
        (52, "Teriyaki Glaze",                "sauces",         5.49, "Sweet and salty glaze for bowls, tofu, and grilled meat",             0),
        (53, "Tomato Basil Pasta Sauce",      "sauces",         5.79, "Bright tomato sauce with fresh basil notes",                          0),
        (54, "Salsa Verde",                   "sauces",         4.79, "Tangy green salsa with tomatillo and cilantro",                       0),
        (55, "Organic Tahini Sauce",          "sauces",         6.99, "Creamy organic sesame sauce with lemon and garlic",                   1),
        (56, "Hot Sauce",                     "sauces",         3.99, "Spicy chili pepper sauce for everyday heat",                          0),
        # --- Spices (8) ---
        (57, "Organic Turmeric Powder",       "spices",         7.49, "Golden organic turmeric with warm earthy notes",                      1),
        (58, "Ground Cinnamon",               "spices",         4.99, "Sweet aromatic cinnamon for baking and drinks",                       0),
        (59, "Cumin Seeds",                   "spices",         3.99, "Toasty cumin seeds with a deep savory aroma",                         0),
        (60, "Smoked Paprika",                "spices",         5.49, "Rich smoked paprika with a warm peppery finish",                      0),
        (61, "Organic Black Pepper",          "spices",         6.49, "Freshly ground organic black pepper with bold aroma",                1),
        (62, "Garlic Powder",                 "spices",         4.49, "Convenient garlic powder for seasoning blends",                       0),
        (63, "Chili Flakes",                  "spices",         3.49, "Crushed red chili flakes for a quick heat boost",                     0),
        (64, "Cardamom Pods",                 "spices",         8.99, "Fragrant cardamom pods for tea, coffee, and desserts",                0),
        # --- Baking (8) ---
        (65, "Almond Flour",                  "baking",         9.99, "Finely ground almond flour for gluten-free baking",                   0),
        (66, "Organic Coconut Sugar",         "baking",         6.99, "Low-refined coconut sugar with caramel notes",                        1),
        (67, "Baking Soda",                   "baking",         2.49, "Kitchen staple for leavening and cleaning",                           0),
        (68, "Vanilla Extract",               "baking",         8.49, "Pure vanilla extract for cookies, cakes, and custards",              0),
        (69, "Organic Cocoa Powder",          "baking",         7.99, "Unsweetened organic cocoa powder with deep chocolate flavor",         1),
        (70, "Baking Powder",                 "baking",         2.29, "Double-acting baking powder for cakes and muffins",                   0),
        (71, "Tapioca Starch",                "baking",         4.29, "Smooth thickener for sauces and gluten-free baking",                  0),
        (72, "Brown Sugar",                   "baking",         3.99, "Soft brown sugar for baking and sweet sauces",                        0),
        # --- Canned (8) ---
        (73, "Organic Black Beans",           "canned",         2.49, "Tender organic black beans ready for quick meals",                    1),
        (74, "Chickpeas",                     "canned",         2.29, "Pantry-friendly chickpeas for salads and hummus",                     0),
        (75, "Diced Tomatoes",                "canned",         1.99, "Kitchen-ready diced tomatoes for soups and sauces",                   0),
        (76, "Coconut Cream",                 "canned",         3.49, "Rich coconut cream for curries and desserts",                         0),
        (77, "Organic Lentils",               "canned",         2.79, "Organic lentils in a convenient ready-to-use can",                    1),
        (78, "Sweet Corn",                    "canned",         1.79, "Sweet whole kernel corn for sides and casseroles",                    0),
        (79, "Kidney Beans",                  "canned",         2.19, "Hearty kidney beans for chili and stews",                             0),
        (80, "Pumpkin Puree",                 "canned",         3.29, "Smooth pumpkin puree for pies, soups, and baking",                    0),
        # --- Beverages (8) ---
        (81, "Sparkling Water",               "beverages",      1.49, "Lightly carbonated water with a crisp finish",                        0),
        (82, "Cold Brew Coffee",              "beverages",      4.99, "Smooth cold brew coffee concentrate, ready to pour",                  0),
        (83, "Kombucha",                      "beverages",      3.99, "Fermented tea beverage with a tangy probiotic profile",              0),
        (84, "Aloe Vera Drink",               "beverages",      2.99, "Refreshing aloe drink with a lightly sweet taste",                    0),
        (85, "Coconut Water",                 "beverages",      2.49, "Naturally hydrating coconut water with electrolytes",                 0),
        (86, "Herbal Tonic",                  "beverages",      3.49, "Botanical tonic with ginger and citrus notes",                        0),
        (87, "Oat Latte",                     "beverages",      4.49, "Creamy oat-based latte for a smooth coffee break",                    0),
        (88, "Green Juice",                   "beverages",      5.99, "Fresh pressed green juice with cucumber and apple",                   0),
        # --- Personal Care (8) ---
        (89, "Organic Shampoo",               "personal-care",   8.99, "Gentle organic shampoo with botanical extracts",                      1),
        (90, "Toothpaste",                    "personal-care",   4.99, "Mint toothpaste for daily oral care",                                 0),
        (91, "Body Wash",                     "personal-care",   7.49, "Refreshing body wash with a clean citrus scent",                     0),
        (92, "Deodorant",                     "personal-care",   6.49, "Long-lasting deodorant with a fresh scent",                           0),
        (93, "Face Cleanser",                 "personal-care",   9.49, "Daily facial cleanser for a gentle deep clean",                       0),
        (94, "Hand Soap",                     "personal-care",   4.49, "Foaming hand soap with a mild lavender scent",                        0),
        (95, "Lip Balm",                      "personal-care",   3.29, "Moisturizing lip balm for everyday use",                              0),
        (96, "Sunscreen",                     "personal-care",  12.99, "Broad-spectrum sunscreen with lightweight feel",                      0),
        # --- Household (4) ---
        (97, "Dish Soap",                     "household",       3.99, "Grease-cutting dish soap with fresh scent",                            0),
        (98, "Laundry Detergent",             "household",      14.99, "High-efficiency laundry detergent for everyday loads",               0),
        (99, "Paper Towels",                  "household",       7.99, "Strong absorbent paper towels for kitchen cleanup",                   0),
        (100, "Reusable Storage Bags",        "household",       9.99, "Reusable food storage bags for pantry and freezer use",               0),
    ]
    cursor.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?)", products)

    # Rebuild FTS index from products
    cursor.execute("INSERT INTO products_fts(products_fts) VALUES ('rebuild')")

    # Build embeddings for all products
    cursor.execute("DELETE FROM product_embeddings")
    rows = cursor.execute(
        "SELECT id, name, category, description FROM products"
    ).fetchall()

    texts = [
        build_product_text(name, category, description)
        for _, name, category, description in rows
    ]
    vectors = EMBED_MODEL.encode(texts, normalize_embeddings=True)

    cursor.executemany(
        "INSERT OR REPLACE INTO product_embeddings (product_id, embedding) VALUES (?, ?)",
        [
            (row[0], serialize_float32(np.asarray(vec, dtype=np.float32)))
            for row, vec in zip(rows, vectors)
        ],
    )

    reviews = [
        # Honey
        (1, 3.8, "Alice",   "Amazing honey! Best I've ever tried."),
        (1, 4.6, "Bob",     "Good quality, will buy again."),
        (1, 4.4, "Carol",   "Excellent raw flavor, very pure."),
        (1, 4.7, "Dave",    "Very good, love that it's unfiltered."),
        (2, 3.2, "Eve",     "Decent honey for the price."),
        (2, 4.1, "Frank",   "Average, nothing special."),
        (2, 4.0, "Grace",   "Good everyday honey."),
        (3, 4.2, "Henry",   "Worth every penny, incredible quality."),
        (3, 4.7, "Iris",    "Excellent antibacterial properties."),
        (3, 4.1, "Jack",    "Best honey I have ever tasted."),
        (4, 4.7, "Kate",    "Okay for cooking, nothing fancy."),
        (4, 4.2, "Leo",     "Nothing special, pretty generic."),
        (4, 3.9, "Mia",     "Average clover honey."),
        (5, 5.0, "Noah",    "Rich bold flavor, great in tea."),
        (5, 4.2, "Olivia",  "Good strong honey, unique taste."),
        (5, 4.3, "Paul",    "Love the dark color and depth."),
        (5, 4.8, "Quinn",   "Great organic option at this price."),
        (6, 3.8, "Rachel",  "Nice floral flavor."),
        (6, 4.0, "Sam",     "Lovely and delicate."),
        (6, 3.8, "Tina",    "Good for baking."),
        (7, 3.8, "Uma",     "Perfect mild flavor, love it!"),
        (7, 4.5, "Victor",  "Excellent light honey."),
        (7, 4.1, "Wendy",   "Great product, very pure taste."),
        (7, 3.3, "Xavier",  "Wonderful, highly recommend."),
        (8, 4.2, "Yvonne",  "Nice spreadable texture."),
        (8, 4.4, "Zack",    "Good on toast."),
        (8, 4.7, "Amy",     "Decent creamed honey."),
        # Oils
        (9,  3.0, "Brian",  "Best olive oil I've used, very fresh."),
        (9,  3.5, "Clara",  "Great flavor, organic certified."),
        (9,  4.5, "Derek",  "Excellent quality, love it."),
        (10, 3.5, "Elena",  "Good for frying, neutral taste."),
        (10, 3.9, "Felix",  "Does the job, nothing exciting."),
        (10, 4.6, "Gina",   "Decent but slightly greasy."),
        (11, 4.8, "Harry",  "Great for smoothies, very fresh."),
        (11, 3.3, "Isla",   "Good omega-3 source, mild flavor."),
        (11, 4.1, "James",  "Love this for salad dressings."),
        (12, 3.6, "Karen",  "Excellent smoke point, tastes great."),
        (12, 4.4, "Liam",   "Good all-purpose oil."),
        (12, 4.4, "Maya",   "Great for cooking and salads."),
        # Nuts & Seeds
        (13, 3.0, "Nate",   "Crunchy and fresh, great snack."),
        (13, 3.6, "Olivia", "Love that they're organic and raw."),
        (13, 4.1, "Peter",  "Perfect size, very fresh."),
        (13, 3.6, "Rita",   "Best almonds I've bought online."),
        (14, 4.6, "Steve",  "Good cashews, nice crunch."),
        (14, 4.4, "Tara",   "Tasty but slightly over-salted."),
        (14, 4.4, "Ursula", "Good value for the quantity."),
        (15, 4.1, "Vince",  "Easy to add to smoothies, love it."),
        (15, 3.9, "Wanda",  "Great fiber source, very fresh."),
        (15, 4.0, "Xena",   "Good quality organic chia seeds."),
        (16, 4.8, "Yuri",   "Good mix, well-balanced variety."),
        (16, 4.5, "Zara",   "A bit too many peanuts for my taste."),
        (16, 5.0, "Alex",   "Nice mix for snacking."),
        (16, 4.9, "Blake",  "Would prefer fewer raisins."),
        # Grains
        (17, 4.9, "Chloe",  "Cooks perfectly, great nutty flavor."),
        (17, 2.9, "Dylan",  "Excellent protein content, love it."),
        (17, 4.8, "Ella",   "Best quinoa I've tried."),
        (18, 3.6, "Finn",   "Great oats, cook quickly and evenly."),
        (18, 4.2, "Gabi",   "Good quality, nice texture."),
        (18, 4.8, "Hugo",   "Reliable everyday oats."),
        (19, 4.9, "Irene",  "Nice chewy texture, great organic choice."),
        (19, 3.9, "Jake",   "Good quality, cooks evenly."),
        (19, 5.0, "Kara",   "Love the organic certification."),
        (20, 4.1, "Lars",   "Great texture, a bit longer to cook."),
        (20, 4.1, "Mona",   "Takes forever to cook but tastes good."),
        (20, 4.5, "Ned",    "Hearty and filling."),
        # Tea & Coffee
        (21, 4.0, "Opal",   "Delicate flavor, very smooth."),
        (21, 4.2, "Phil",   "Best green tea I've had, very fresh."),
        (21, 4.4, "Quinn",  "Great quality, calming and tasty."),
        (22, 4.7, "Rose",   "Very soothing before bed."),
        (22, 3.9, "Seth",   "Lovely floral notes, very relaxing."),
        (22, 3.7, "Tess",   "Good chamomile, nice and mild."),
        (23, 4.7, "Uri",    "Best coffee I've ever brewed at home."),
        (23, 4.8, "Vera",   "Amazing single-origin flavor."),
        (23, 4.3, "Will",   "Very smooth with great aroma."),
        (23, 4.0, "Xara",   "Exceptional quality, worth every penny."),
        (24, 4.8, "Yael",   "Strong and bold, perfect espresso."),
        (24, 3.4, "Zion",   "Good dark roast, consistent grind."),
        (24, 3.0, "Abe",    "Solid everyday espresso blend."),
        # Snacks
        (25, 3.7, "Beth",   "Delicious and not too sweet."),
        (25, 3.5, "Cole",   "Great texture, love the almonds."),
        (25, 4.9, "Dana",   "My go-to breakfast granola."),
        (26, 3.0, "Earl",   "Light and crispy, good for dieting."),
        (26, 3.0, "Faye",   "A bit bland but does the job."),
        (26, 4.1, "Glen",   "Good value, decent snack."),
        (27, 4.8, "Hope",   "So sweet and chewy, love these!"),
        (27, 4.5, "Ivan",   "Great that there's no added sugar."),
        (27, 4.5, "Jade",   "Perfect snack, very natural taste."),
        (28, 4.4, "Kent",   "Good mix, great for hiking."),
        (28, 3.6, "Luna",   "Too many M&Ms, prefer less candy."),
        (28, 4.5, "Marc",   "Decent but not my favorite mix."),
        # Dairy Alternatives
        (29, 3.9, "Nina",   "Great in coffee, smooth texture."),
        (29, 4.7, "Omar",   "Love the organic certification."),
        (29, 4.5, "Pam",    "Tastes great and not too thin."),
        (30, 5.0, "Rex",    "Perfect for lattes, froths well."),
        (30, 3.7, "Sara",   "Good oat milk, slightly sweet."),
        (30, 3.5, "Tom",    "Best barista oat milk I've tried."),
        (31, 4.7, "Una",    "Creamy and rich, great for curries."),
        (31, 4.6, "Vito",   "Full fat and delicious."),
        (31, 3.2, "Wren",   "Perfect coconut milk, great quality."),
        (32, 4.5, "Xio",    "Good protein content, mild flavor."),
        (32, 4.0, "Yosef",  "Slightly thin but good for cereal."),
        (32, 4.1, "Zola",   "Decent soy milk, nothing special."),
        # Superfoods
        (33, 4.7, "Ava1", "Organic Spirulina Powder has great quality and arrived fresh."),
        (33, 4.5, "Hugo2", "Solid value for a superfoods item; I would buy it again."),
        (33, 3.9, "Owen3", "I like using Organic Spirulina Powder in my regular routine."),
        (34, 4.0, "Ben1", "Cacao Nibs has great quality and arrived fresh."),
        (34, 4.6, "Ivy2", "Solid value for a superfoods item; I would buy it again."),
        (34, 2.8, "Pia3", "I like using Cacao Nibs in my regular routine."),
        (35, 3.3, "Cora1", "Organic Maca Powder has great quality and arrived fresh."),
        (35, 4.4, "Jonah2", "Solid value for a superfoods item; I would buy it again."),
        (35, 4.0, "Quinn3", "I like using Organic Maca Powder in my regular routine."),
        (36, 3.6, "Diego1", "Hemp Hearts has great quality and arrived fresh."),
        (36, 4.8, "Kira2", "Solid value for a superfoods item; I would buy it again."),
        (36, 4.8, "Rosa3", "I like using Hemp Hearts in my regular routine."),
        (37, 4.1, "Emma1", "Goji Berries has great quality and arrived fresh."),
        (37, 3.3, "Leo2", "Solid value for a superfoods item; I would buy it again."),
        (37, 4.7, "Sage3", "I like using Goji Berries in my regular routine."),
        (38, 3.0, "Finn1", "Organic Matcha Powder has great quality and arrived fresh."),
        (38, 3.1, "Mila2", "Solid value for a superfoods item; I would buy it again."),
        (38, 3.8, "Theo3", "I like using Organic Matcha Powder in my regular routine."),
        (39, 2.8, "Gia1", "Pumpkin Seeds has great quality and arrived fresh."),
        (39, 4.7, "Noah2", "Solid value for a superfoods item; I would buy it again."),
        (39, 4.2, "Uma3", "I like using Pumpkin Seeds in my regular routine."),
        (40, 4.4, "Hugo1", "Acai Powder has great quality and arrived fresh."),
        (40, 4.4, "Owen2", "Solid value for a superfoods item; I would buy it again."),
        (40, 3.8, "Vera3", "I like using Acai Powder in my regular routine."),
        # Pasta
        (41, 4.8, "Ivy1", "Organic Spaghetti has great quality and arrived fresh."),
        (41, 4.8, "Pia2", "Solid value for a pasta item; I would buy it again."),
        (41, 4.5, "Wes3", "I like using Organic Spaghetti in my regular routine."),
        (42, 4.3, "Jonah1", "Penne Rigate has great quality and arrived fresh."),
        (42, 3.8, "Quinn2", "Solid value for a pasta item; I would buy it again."),
        (42, 3.5, "Xena3", "I like using Penne Rigate in my regular routine."),
        (43, 3.9, "Kira1", "Chickpea Pasta has great quality and arrived fresh."),
        (43, 4.6, "Rosa2", "Solid value for a pasta item; I would buy it again."),
        (43, 3.8, "Yara3", "I like using Chickpea Pasta in my regular routine."),
        (44, 4.9, "Leo1", "Brown Rice Pasta has great quality and arrived fresh."),
        (44, 4.2, "Sage2", "Solid value for a pasta item; I would buy it again."),
        (44, 4.2, "Zane3", "I like using Brown Rice Pasta in my regular routine."),
        (45, 3.9, "Mila1", "Organic Fusilli has great quality and arrived fresh."),
        (45, 3.4, "Theo2", "Solid value for a pasta item; I would buy it again."),
        (45, 3.7, "Ava3", "I like using Organic Fusilli in my regular routine."),
        (46, 4.5, "Noah1", "Lasagna Sheets has great quality and arrived fresh."),
        (46, 4.7, "Uma2", "Solid value for a pasta item; I would buy it again."),
        (46, 5.0, "Ben3", "I like using Lasagna Sheets in my regular routine."),
        (47, 4.2, "Owen1", "Gluten-Free Lentil Pasta has great quality and arrived fresh."),
        (47, 4.8, "Vera2", "Solid value for a pasta item; I would buy it again."),
        (47, 4.5, "Cora3", "I like using Gluten-Free Lentil Pasta in my regular routine."),
        (48, 5.0, "Pia1", "Whole Wheat Macaroni has great quality and arrived fresh."),
        (48, 4.0, "Wes2", "Solid value for a pasta item; I would buy it again."),
        (48, 4.1, "Diego3", "I like using Whole Wheat Macaroni in my regular routine."),
        # Sauces
        (49, 5.0, "Quinn1", "Organic Marinara Sauce has great quality and arrived fresh."),
        (49, 4.2, "Xena2", "Solid value for a sauces item; I would buy it again."),
        (49, 4.9, "Emma3", "I like using Organic Marinara Sauce in my regular routine."),
        (50, 3.4, "Rosa1", "Alfredo Sauce has great quality and arrived fresh."),
        (50, 4.8, "Yara2", "Solid value for a sauces item; I would buy it again."),
        (50, 3.3, "Finn3", "I like using Alfredo Sauce in my regular routine."),
        (51, 4.9, "Sage1", "Organic Peanut Sauce has great quality and arrived fresh."),
        (51, 4.6, "Zane2", "Solid value for a sauces item; I would buy it again."),
        (51, 4.6, "Gia3", "I like using Organic Peanut Sauce in my regular routine."),
        (52, 4.6, "Theo1", "Teriyaki Glaze has great quality and arrived fresh."),
        (52, 5.0, "Ava2", "Solid value for a sauces item; I would buy it again."),
        (52, 4.7, "Hugo3", "I like using Teriyaki Glaze in my regular routine."),
        (53, 4.6, "Uma1", "Tomato Basil Pasta Sauce has great quality and arrived fresh."),
        (53, 3.2, "Ben2", "Solid value for a sauces item; I would buy it again."),
        (53, 4.4, "Ivy3", "I like using Tomato Basil Pasta Sauce in my regular routine."),
        (54, 4.9, "Vera1", "Salsa Verde has great quality and arrived fresh."),
        (54, 3.9, "Cora2", "Solid value for a sauces item; I would buy it again."),
        (54, 4.7, "Jonah3", "I like using Salsa Verde in my regular routine."),
        (55, 4.6, "Wes1", "Organic Tahini Sauce has great quality and arrived fresh."),
        (55, 4.8, "Diego2", "Solid value for a sauces item; I would buy it again."),
        (55, 4.1, "Kira3", "I like using Organic Tahini Sauce in my regular routine."),
        (56, 3.8, "Xena1", "Hot Sauce has great quality and arrived fresh."),
        (56, 4.7, "Emma2", "Solid value for a sauces item; I would buy it again."),
        (56, 3.8, "Leo3", "I like using Hot Sauce in my regular routine."),
        # Spices
        (57, 4.5, "Yara1", "Organic Turmeric Powder has great quality and arrived fresh."),
        (57, 4.6, "Finn2", "Solid value for a spices item; I would buy it again."),
        (57, 4.4, "Mila3", "I like using Organic Turmeric Powder in my regular routine."),
        (58, 4.6, "Zane1", "Ground Cinnamon has great quality and arrived fresh."),
        (58, 4.5, "Gia2", "Solid value for a spices item; I would buy it again."),
        (58, 4.9, "Noah3", "I like using Ground Cinnamon in my regular routine."),
        (59, 4.6, "Ava1", "Cumin Seeds has great quality and arrived fresh."),
        (59, 4.5, "Hugo2", "Solid value for a spices item; I would buy it again."),
        (59, 3.6, "Owen3", "I like using Cumin Seeds in my regular routine."),
        (60, 3.6, "Ben1", "Smoked Paprika has great quality and arrived fresh."),
        (60, 4.8, "Ivy2", "Solid value for a spices item; I would buy it again."),
        (60, 4.9, "Pia3", "I like using Smoked Paprika in my regular routine."),
        (61, 3.9, "Cora1", "Organic Black Pepper has great quality and arrived fresh."),
        (61, 3.9, "Jonah2", "Solid value for a spices item; I would buy it again."),
        (61, 4.3, "Quinn3", "I like using Organic Black Pepper in my regular routine."),
        (62, 3.0, "Diego1", "Garlic Powder has great quality and arrived fresh."),
        (62, 4.5, "Kira2", "Solid value for a spices item; I would buy it again."),
        (62, 3.5, "Rosa3", "I like using Garlic Powder in my regular routine."),
        (63, 3.4, "Emma1", "Chili Flakes has great quality and arrived fresh."),
        (63, 4.6, "Leo2", "Solid value for a spices item; I would buy it again."),
        (63, 3.4, "Sage3", "I like using Chili Flakes in my regular routine."),
        (64, 4.5, "Finn1", "Cardamom Pods has great quality and arrived fresh."),
        (64, 4.4, "Mila2", "Solid value for a spices item; I would buy it again."),
        (64, 4.9, "Theo3", "I like using Cardamom Pods in my regular routine."),
        (65, 3.1, "Gia1", "Almond Flour has great quality and arrived fresh."),
        (65, 3.2, "Noah2", "Solid value for a baking item; I would buy it again."),
        (65, 4.8, "Uma3", "I like using Almond Flour in my regular routine."),
        (66, 4.8, "Hugo1", "Organic Coconut Sugar has great quality and arrived fresh."),
        (66, 3.8, "Owen2", "Solid value for a baking item; I would buy it again."),
        (66, 4.8, "Vera3", "I like using Organic Coconut Sugar in my regular routine."),
        (67, 3.2, "Ivy1", "Baking Soda has great quality and arrived fresh."),
        (67, 4.9, "Pia2", "Solid value for a baking item; I would buy it again."),
        (67, 4.6, "Wes3", "I like using Baking Soda in my regular routine."),
        (68, 4.3, "Jonah1", "Vanilla Extract has great quality and arrived fresh."),
        (68, 3.3, "Quinn2", "Solid value for a baking item; I would buy it again."),
        (68, 4.7, "Xena3", "I like using Vanilla Extract in my regular routine."),
        (69, 4.8, "Kira1", "Organic Cocoa Powder has great quality and arrived fresh."),
        (69, 4.0, "Rosa2", "Solid value for a baking item; I would buy it again."),
        (69, 3.9, "Yara3", "I like using Organic Cocoa Powder in my regular routine."),
        (70, 3.7, "Leo1", "Baking Powder has great quality and arrived fresh."),
        (70, 5.0, "Sage2", "Solid value for a baking item; I would buy it again."),
        (70, 3.3, "Zane3", "I like using Baking Powder in my regular routine."),
        (71, 2.9, "Mila1", "Tapioca Starch has great quality and arrived fresh."),
        (71, 3.3, "Theo2", "Solid value for a baking item; I would buy it again."),
        (71, 4.8, "Ava3", "I like using Tapioca Starch in my regular routine."),
        (72, 3.9, "Noah1", "Brown Sugar has great quality and arrived fresh."),
        (72, 3.7, "Uma2", "Solid value for a baking item; I would buy it again."),
        (72, 3.8, "Ben3", "I like using Brown Sugar in my regular routine."),
        # Canned
        (73, 4.1, "Owen1", "Organic Black Beans has great quality and arrived fresh."),
        (73, 3.3, "Vera2", "Solid value for a canned item; I would buy it again."),
        (73, 4.4, "Cora3", "I like using Organic Black Beans in my regular routine."),
        (74, 3.4, "Pia1", "Chickpeas has great quality and arrived fresh."),
        (74, 3.8, "Wes2", "Solid value for a canned item; I would buy it again."),
        (74, 3.5, "Diego3", "I like using Chickpeas in my regular routine."),
        (75, 4.6, "Quinn1", "Diced Tomatoes has great quality and arrived fresh."),
        (75, 3.2, "Xena2", "Solid value for a canned item; I would buy it again."),
        (75, 4.8, "Emma3", "I like using Diced Tomatoes in my regular routine."),
        (76, 4.1, "Rosa1", "Coconut Cream has great quality and arrived fresh."),
        (76, 4.6, "Yara2", "Solid value for a canned item; I would buy it again."),
        (76, 3.9, "Finn3", "I like using Coconut Cream in my regular routine."),
        (77, 4.1, "Sage1", "Organic Lentils has great quality and arrived fresh."),
        (77, 3.9, "Zane2", "Solid value for a canned item; I would buy it again."),
        (77, 4.6, "Gia3", "I like using Organic Lentils in my regular routine."),
        (78, 3.8, "Theo1", "Sweet Corn has great quality and arrived fresh."),
        (78, 3.8, "Ava2", "Solid value for a canned item; I would buy it again."),
        (78, 5.0, "Hugo3", "I like using Sweet Corn in my regular routine."),
        (79, 4.5, "Uma1", "Kidney Beans has great quality and arrived fresh."),
        (79, 4.3, "Ben2", "Solid value for a canned item; I would buy it again."),
        (79, 4.0, "Ivy3", "I like using Kidney Beans in my regular routine."),
        (80, 4.3, "Vera1", "Pumpkin Puree has great quality and arrived fresh."),
        (80, 4.6, "Cora2", "Solid value for a canned item; I would buy it again."),
        (80, 4.5, "Jonah3", "I like using Pumpkin Puree in my regular routine."),
        # Beverages
        (81, 3.4, "Wes1", "Sparkling Water has great quality and arrived fresh."),
        (81, 4.9, "Diego2", "Solid value for a beverages item; I would buy it again."),
        (81, 4.7, "Kira3", "I like using Sparkling Water in my regular routine."),
        (82, 4.8, "Xena1", "Cold Brew Coffee has great quality and arrived fresh."),
        (82, 4.4, "Emma2", "Solid value for a beverages item; I would buy it again."),
        (82, 4.2, "Leo3", "I like using Cold Brew Coffee in my regular routine."),
        (83, 4.7, "Yara1", "Kombucha has great quality and arrived fresh."),
        (83, 4.3, "Finn2", "Solid value for a beverages item; I would buy it again."),
        (83, 4.7, "Mila3", "I like using Kombucha in my regular routine."),
        (84, 3.8, "Zane1", "Aloe Vera Drink has great quality and arrived fresh."),
        (84, 3.9, "Gia2", "Solid value for a beverages item; I would buy it again."),
        (84, 3.2, "Noah3", "I like using Aloe Vera Drink in my regular routine."),
        (85, 4.6, "Ava1", "Coconut Water has great quality and arrived fresh."),
        (85, 3.6, "Hugo2", "Solid value for a beverages item; I would buy it again."),
        (85, 4.1, "Owen3", "I like using Coconut Water in my regular routine."),
        (86, 4.4, "Ben1", "Herbal Tonic has great quality and arrived fresh."),
        (86, 4.7, "Ivy2", "Solid value for a beverages item; I would buy it again."),
        (86, 3.0, "Pia3", "I like using Herbal Tonic in my regular routine."),
        (87, 4.0, "Cora1", "Oat Latte has great quality and arrived fresh."),
        (87, 3.8, "Jonah2", "Solid value for a beverages item; I would buy it again."),
        (87, 4.3, "Quinn3", "I like using Oat Latte in my regular routine."),
        (88, 3.8, "Diego1", "Green Juice has great quality and arrived fresh."),
        (88, 5.0, "Kira2", "Solid value for a beverages item; I would buy it again."),
        (88, 4.6, "Rosa3", "I like using Green Juice in my regular routine."),
        # Personal Care
        (89, 4.3, "Emma1", "Organic Shampoo has great quality and arrived fresh."),
        (89, 4.4, "Leo2", "Solid value for a personal-care item; I would buy it again."),
        (89, 4.1, "Sage3", "I like using Organic Shampoo in my regular routine."),
        (90, 3.2, "Finn1", "Toothpaste has great quality and arrived fresh."),
        (90, 3.8, "Mila2", "Solid value for a personal-care item; I would buy it again."),
        (90, 3.8, "Theo3", "I like using Toothpaste in my regular routine."),
        (91, 4.5, "Gia1", "Body Wash has great quality and arrived fresh."),
        (91, 3.6, "Noah2", "Solid value for a personal-care item; I would buy it again."),
        (91, 3.7, "Uma3", "I like using Body Wash in my regular routine."),
        (92, 5.0, "Hugo1", "Deodorant has great quality and arrived fresh."),
        (92, 4.6, "Owen2", "Solid value for a personal-care item; I would buy it again."),
        (92, 4.8, "Vera3", "I like using Deodorant in my regular routine."),
        (93, 4.6, "Ivy1", "Face Cleanser has great quality and arrived fresh."),
        (93, 4.4, "Pia2", "Solid value for a personal-care item; I would buy it again."),
        (93, 4.9, "Wes3", "I like using Face Cleanser in my regular routine."),
        (94, 4.3, "Jonah1", "Hand Soap has great quality and arrived fresh."),
        (94, 4.7, "Quinn2", "Solid value for a personal-care item; I would buy it again."),
        (94, 3.2, "Xena3", "I like using Hand Soap in my regular routine."),
        (95, 4.3, "Kira1", "Lip Balm has great quality and arrived fresh."),
        (95, 3.8, "Rosa2", "Solid value for a personal-care item; I would buy it again."),
        (95, 3.2, "Yara3", "I like using Lip Balm in my regular routine."),
        (96, 4.7, "Leo1", "Sunscreen has great quality and arrived fresh."),
        (96, 2.9, "Sage2", "Solid value for a personal-care item; I would buy it again."),
        (96, 3.9, "Zane3", "I like using Sunscreen in my regular routine."),
        # Household
        (97, 3.4, "Mila1", "Dish Soap has great quality and arrived fresh."),
        (97, 3.8, "Theo2", "Solid value for a household item; I would buy it again."),
        (97, 4.4, "Ava3", "I like using Dish Soap in my regular routine."),
        (98, 4.7, "Noah1", "Laundry Detergent has great quality and arrived fresh."),
        (98, 4.5, "Uma2", "Solid value for a household item; I would buy it again."),
        (98, 5.0, "Ben3", "I like using Laundry Detergent in my regular routine."),
        (99, 4.4, "Owen1", "Paper Towels has great quality and arrived fresh."),
        (99, 4.0, "Vera2", "Solid value for a household item; I would buy it again."),
        (99, 4.4, "Cora3", "I like using Paper Towels in my regular routine."),
        (100, 4.6, "Pia1", "Reusable Storage Bags has great quality and arrived fresh."),
        (100, 3.9, "Wes2", "Solid value for a household item; I would buy it again."),
        (100, 4.3, "Diego3", "I like using Reusable Storage Bags in my regular routine."),
    ]
    cursor.execute("DELETE FROM reviews")
    cursor.executemany(
        "INSERT INTO reviews (product_id, rating, reviewer_name, review_text) VALUES (?, ?, ?, ?)",
        reviews,
    )

    conn.commit()
    conn.close()
    print(f"Database created at: {DB_PATH}")


if __name__ == "__main__":
    create_database()
