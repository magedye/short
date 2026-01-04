

python main.py
streamlit run ui.py

streamlit run ui.py

# 1. Build the images
docker-compose build

# 2. Start the services in the background
docker-compose up -d

# 3. Check if containers are running
docker ps



# 1. إيقاف وإزالة الحاويات والشبكات القديمة تماماً
docker-compose down

# 2. بناء الصور من جديد بدون استخدام الـ Cache (لضمان تطبيق تعديلات libaio أو pip)
docker-compose build --no-cache

# 3. التشغيل في الخلفية
docker-compose up -d

# 4. مراقبة سجل الأخطاء فوراً للتأكد من نجاح التشغيل
docker-compose logs -f backend


python - <<'PY'
import os, json, chromadb
from pprint import pprint

chroma_path = os.getenv("CHROMA_PATH", "./vanna_memory")
collection_name = os.getenv("CHROMA_COLLECTION", "easydata_memory")

client = chromadb.PersistentClient(path=chroma_path)
coll = client.get_collection(collection_name)

print(f"Collection: {collection_name} | Path: {chroma_path} | Count: {coll.count()}")

res = coll.get(limit=50)  # عدل limit إذا احتجت
docs = []
for i, _id in enumerate(res.get("ids", [])):
    docs.append({
        "id": _id,
        "metadata": res.get("metadatas", [{}])[i],
        "doc": res.get("documents", [""])[i][:500],  # قصّ المحتوى لسهولة المعاينة
    })

pprint(docs)
PY
