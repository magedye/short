

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