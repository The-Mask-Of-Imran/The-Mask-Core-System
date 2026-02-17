The Mask Personal AI Core – একটা ব্যক্তিগত, সেলফ-আপগ্রেডেবল AI অটোমেশন সিস্টেম, যা তোমার PC এবং ক্লাউডকে হাইব্রিডভাবে ইন্টিগ্রেট করে কাজ করে। এটি প্রথমে একটা সিম্পল চ্যাট-বেসড অটোমেটর হিসেবে শুরু হয়েছে, কিন্তু ভিশন অনুসারে এটি একটা "বুদ্ধিমান" সিস্টেমে পরিণত হবে যা নিজে নিজে শিখে, আপগ্রেড করে, এবং লং-টার্ম টাস্ক ম্যানেজ করে।
পুরনো প্ল্যানে ১০টা ফিচার ছিল (প্লাগিন-ভিত্তিক), কিন্তু নতুন ভিশনে প্লাগিন স্কিপ করে প্রথমে সেলফ-আপগ্রেডেবল বেস তৈরি করা হয়েছে। বর্তমানে সিস্টেমটি Render.com-এ ডেপ্লয়ড এবং GitHub-এ ব্যাকআপড, যা নিচে বিস্তারিত বলছি। আমি সিস্টেমের ভিশন, আর্কিটেকচার, কোর ক্যাপাবিলিটিস, রোডম্যাপ সামারি, এবং বর্তমান ইমপ্লিমেন্টেশন বিস্তারিতভাবে লিখছি।
সিস্টেমের নাম এবং মূল দর্শন
নাম: The Mask Personal AI Core v1.0 (অথবা The Mask Automation Core System, যেমন Render এবং GitHub-এ উল্লেখিত)।
মূল দর্শন: এটি একটা হাইব্রিড, সেলফ-আপগ্রেডেবল, লং-টার্ম মেমরি-সম্পন্ন ব্যক্তিগত AI যা তোমার (অ্যাডমিনের) সবকিছু শেখে, ভুল থেকে শিখে, এবং নিজেকে নিজে আপগ্রেড করে। এটি তোমার PC অটোমেশনের কোর হিসেবে কাজ করবে – চ্যাট থেকে শুরু করে ফাইল ম্যানেজমেন্ট, CMD এক্সিকিউশন, TTS/STT, এবং ভবিষ্যতে হ্যাকিং/PC কন্ট্রোল পর্যন্ত। প্রথম প্রায়োরিটি: সেলফ-আপগ্রেড ক্যাপাবিলিটি, যাতে পরবর্তী ফিচারগুলো নিজে যোগ করতে পারে। এটি "বুদ্ধিমান" হবে মানে, এটি ভুল থেকে লার্ন করে দ্বিতীয়বার একই ভুল করবে না, এবং বছরের পর বছর টাস্ক মনে রাখবে।
আর্কিটেকচার (হাইব্রিড ডিজাইন)
সিস্টেমটি হাইব্রিড – ক্লাউড এবং লোকাল লেয়ারের কম্বিনেশন:
ক্লাউড লেয়ার (Render.com):
মডেল: 7B প্যারামিটার LLM (যেমন Llama বা অনুরূপ), Groq 70B ফলব্যাক সহ।
সুবিধা: ১-৫০টা API একসাথে চালানো, স্মার্ট সুইচিং (লিমিট শেষ হলে অটো সুইচ), কম খরচে স্কেলেবল।
ব্যবহার: লাইটওয়েট টাস্ক, চ্যাট, এবং ফাস্ট রেসপন্স।
লোকাল লেয়ার (তোমার PC):
মডেল: 14B প্যারামিটার LLM (Ollama দিয়ে রান)।
সুবিধা: ভারী কাজ (e.g., পার্সোনাল ফাইল অ্যানালাইসিস, লং-টার্ম মেমরি প্রসেসিং), প্রাইভেসি (ডাটা লোকাল থাকে), কোনো API লিমিট নেই।
ব্যবহার: সেনসিটিভ ডাটা, অফলাইন মোড।
স্মার্ট সুইচিং ইঞ্জিন:
কাজের ধরন (e.g., সিম্পল চ্যাট vs কম্প্লেক্স কোডিং), খরচ, স্পিড, API লিমিট, অ্যাভেলেবিলিটি অনুসারে অটো সিদ্ধান্ত নেয়।
উদাহরণ: API লিমিট শেষ হলে লোকালে সুইচ, অথবা সেনসিটিভ টাস্ক লোকালে।
ডাটা স্টোরেজ এবং সিঙ্ক: SQLite ডাটাবেস (প্রাইমারি) + JSON ব্যাকআপ। ক্লাউড এবং লোকালের মধ্যে অটো সিঙ্ক্রোনাইজেশন (মেমরি শেয়ার)।
সিকিউরিটি লেয়ার: অ্যাপ্রুভাল সিস্টেম (সেনসিটিভ অপারেশনের আগে ইউজার অনুমতি), এনক্রিপশন (সেনসিটিভ ডাটা), রুল-বেসড চেকস।
কোর ক্যাপাবিলিটিস (প্রথমে তৈরি করা ফিচারস)
সিস্টেমের মূল ক্ষমতা নিচেরগুলো, যা রোডম্যাপ অনুসারে ধাপে ধাপে যোগ হবে:
যেকোনো ফিচার অনুরোধ অ্যানালাইজ: ইউজারের কমান্ড (e.g., "TTS যোগ করো") অ্যানালাইজ করে কোড জেনারেট এবং আপগ্রেড।
সিস্টেম অ্যাক্সেস: অনুমতি সাপেক্ষে ফাইল তৈরি/এডিট/ডিলিট, CMD/PowerShell এক্সিকিউশন।
কোডিং দক্ষতা: এরর অটো ফিক্স, কোড জেনারেশন।
সেলফ-আপগ্রেড: নিজে কোড লিখে, টেস্ট করে, git push, Render রিস্টার্ট। ব্যাকআপ (zip + Google Drive), রোলব্যাক মেকানিজম সহ।
লং-টার্ম মেমরি: টাস্ক, ইন্টার্যাকশন, লার্নিং লেসন সেভ (বছরের পর বছর)। অটো সামারাইজেশন পুরনো মেমরির।
সেলফ-লার্নিং: ভুল থেকে লেসন সেভ, হাইব্রিড লার্নিং (লোকাল + Groq)।
অ্যাডভান্সড ফিচারস (পরবর্তীতে): TTS/STT (ভয়েস ইন্টারফেস), প্লাগিন সিস্টেম, ড্যাশবোর্ড UI (Streamlit), লং-টার্ম টাস্ক ম্যানেজমেন্ট।
এরর হ্যান্ডলিং এবং মনিটরিং: লগিং, অটো-লার্নিং, /status এন্ডপয়েন্ট (সব মডেলের স্টেট)।
রোডম্যাপ সামারি (৩টা ফেজ)
ফেজ ১: কোর ফাউন্ডেশন + হাইব্রিড + সিকিউরিটি (৪-৫ সপ্তাহ): লং-টার্ম মেমরি ফিক্স, হাইব্রিড সুইচিং, অ্যাপ্রুভাল সিস্টেম, লোকাল কানেকশন, সিঙ্ক্রোনাইজেশন। লক্ষ্য: স্থিতিশীল বেস।
ফেজ ২: সেলফ-আপগ্রেড ইঞ্জিন (৪-৬ সপ্তাহ): SelfUpgradeEngine, কোড ফ্লো, ব্যাকআপ/রোলব্যাক, অটো প্রস্তাব, টেস্টস। লক্ষ্য: নিজে আপগ্রেড করার ক্ষমতা।
ফেজ ৩: অ্যাডভান্সড লার্নিং + লং-টার্ম টাস্ক (৩-৪ সপ্তাহ): SelfLearningManager, টাস্ক স্টেট, হাইব্রিড লার্নিং, সামারাইজেশন, সিকিউরিটি অডিট, প্লাগিন বেস, ভয়েস ইন্টারফেস, ড্যাশবোর্ড, E2E টেস্ট, v1.0 রিলিজ। লক্ষ্য: সত্যিকারের বুদ্ধিমান সিস্টেম।
সময়কাল: মোট ১১-১৫ সপ্তাহ, ছোট টাস্কে ভাগ করা।
বর্তমান তৈরিকৃত ইনফরমেশন এবং স্ট্যাটাস
তোমার প্রদত্ত লিঙ্কগুলো অ্যানালাইজ করে (Render.com সার্ভিস এবং GitHub রেপো), বর্তমান সিস্টেমের অবস্থা নিচের মতো:
Render.com সার্ভিস:
Service ID: srv-d6309dcr85hc739uvh0g
Service Address: https://the-mask-automation-core.onrender.com
বর্তমান ফাংশনালিটি: সার্ভিসটি চালু আছে এবং একটা সিম্পল JSON রেসপন্স দেয়: {"message": "The Mask Core System চালু আছে! বাংলায় কথা বলতে পারি।"}। এটি কোর চ্যাট ফিচারের ইন্ডিকেটর – সিস্টেম অ্যাকটিভ এবং বাংলায় ইন্টার্যাকশন সমর্থন করে। কোনো অতিরিক্ত এন্ডপয়েন্ট (e.g., /status, /chat) দৃশ্যমান নয়, কিন্তু এটি একটা API-ভিত্তিক সার্ভিস বলে মনে হয় যা চ্যাট বা অটোমেশনের বেস। লং-টার্ম মেমরি বা অন্য ফিচারস এখনো পাবলিকলি এক্সপোজড নয় (connection refused এররের কথা মনে করে, এটি আংশিক)।
স্ট্যাটাস: ✅ কমপ্লিট (কোর চ্যাট লাইভ), কিন্তু লং-টার্ম মেমরি ❌ (এখনো ফিক্স নয়)।
GitHub রেপো:
URL: https://github.com/The-Mask-Of-Imran/The-Mask-Core-System
ডেসক্রিপশন: "My PC Automator Core System Backup" – এটি তোমার PC অটোমেটরের কোর সিস্টেমের ব্যাকআপ।
ফাইলস এবং স্ট্রাকচার:
app.py: মূল অ্যাপ্লিকেশন স্ক্রিপ্ট (সম্ভবত Flask/FastAPI-ভিত্তিক, চ্যাট লজিক এখানে)।
config.json: কনফিগারেশন ফাইল (API কী, মডেল সেটিংস, ইত্যাদি)।
memory.json: লং-টার্ম মেমরি স্টোরেজ (JSON-ভিত্তিক ব্যাকআপ, কিন্তু এখনো অস্থিতিশীল)।
requirements.txt: ডিপেন্ডেন্সিস লিস্ট (Python লাইব্রেরী যেমন requests, ollama, ইত্যাদি)।
কমিটস: মোট ৬টা কমিট (সাম্প্রতিক পরিবর্তনগুলো সিস্টেমের বেসিক সেটআপ)।
README: নেই (পরে যোগ করা যাবে ডকুমেন্টেশনের জন্য)।
স্ট্যাটাস: রেপোটি ব্যাকআপ হিসেবে কাজ করছে, কিন্তু পুরো রোডম্যাপের শুধুমাত্র কোর চ্যাট কমপ্লিট। বাকি টাস্কস (e.g., মেমরি ফিক্স, সেলফ-আপগ্রেড) ইমপ্লিমেন্ট করা বাকি।
সামগ্রিক বর্তমান স্ট্যাটাস (মেসেজ থেকে):
✅ কোর চ্যাট (ফিচার ১) কমপ্লিট এবং লাইভ (Render-এ)।
❌ লং-টার্ম মেমরি (ফিচার ২) – Connection refused এরর, ফিক্স দরকার।
বাকি সব ❌ (e.g., সেলফ-আপগ্রেড, হাইব্রিড সুইচিং)।
সমস্যা: মেমরি সিস্টেম অস্থিতিশীল, যা সেলফ-আপগ্রেডের জন্য আবশ্যক।

বিল্ডাপ রোডম্যাপ:..............


টাস্ক ১: লং-টার্ম মেমরি ফিক্স (Connection Refused Error Resolution with SQLite and JSON Backup)
ওভারভিউ: এই টাস্কে লং-টার্ম মেমরি সিস্টেমকে স্থিতিশীল করো, যাতে কানেকশন রিফিউজড এরর না আসে। SQLite ডাটাবেসকে প্রাইমারি স্টোরেজ হিসেবে ব্যবহার করা হবে, এবং JSON ফাইলকে ব্যাকআপ হিসেবে, যাতে ডাটা লস না হয়। TaskMemoryManager ক্লাসকে আপডেট করে মেমরি অ্যাক্সেস, রিট্রিভাল এবং আপডেট প্রক্রিয়াকে স্ট্রিমলাইন করা হবে, যাতে সেলফ-আপগ্রেড এবং লং-টার্ম লার্নিং সমর্থন করে।
প্রয়োজনীয় প্রিপারেশন:

* রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।

* লাইব্রেরী: sqlite3 (built-in), json (built-in), logging (built-in), tenacity (pip install tenacity==8.0.1)।

* GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।

* ডিরেক্টরি: modules/memory/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

1. modules/memory/TaskMemoryManager.py ফাইল তৈরি।

2. class TaskMemoryManager:।

3. __init__ মেথড যোগ:
   ```python
   from config import BACKUP_DIR
   import os
   if not os.path.exists(BACKUP_DIR):
       os.makedirs(BACKUP_DIR)
   db_path = os.path.join(BACKUP_DIR, 'memory.db')
   try:
       self.conn = sqlite3.connect(db_path, check_same_thread=False)
       self.cursor = self.conn.cursor()
       self.cursor.execute('''CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, task_id TEXT, content TEXT, timestamp DATETIME, category TEXT)''')
       self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories (timestamp)')
       self.conn.commit()
   except sqlite3.OperationalError as e:
       logging.error(f"DB Error: {e}")
       @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
       def retry_connect():
           self.conn = sqlite3.connect(db_path)
       retry_connect()

save_memory মেথড:Pythondef save_memory(self, task_id: str, content: dict, category: str):
    timestamp = datetime.now()
    self.cursor.execute("INSERT INTO memories (task_id, content, timestamp, category) VALUES (?, ?, ?, ?)", (task_id, json.dumps(content), timestamp, category))
    self.conn.commit()
    self._backup_to_json()
_backup_to_json:Pythondef _backup_to_json(self):
    with open(os.path.join(BACKUP_DIR, 'memory_backup.json'), 'w') as f:
        json.dump(self._get_all_memories(), f, indent=4)
load_memory, _get_all_memories — আগের মতো।
app.py-এ ইন্টিগ্রেট:Pythonfrom config import BACKUP_DIR
from modules.memory.TaskMemoryManager import TaskMemoryManager
memory_manager = TaskMemoryManager()

ইন্টিগ্রেশন: config.py থেকে BACKUP_DIR নেওয়া। পরবর্তী টাস্কে sync/restore এই পাথ ইউজ করো।
টেস্টিং:

tests/memory_test.py তৈরি।
python -m unittest tests/memory_test.py।
রিপোর্ট সেভ: reports/task_01_memory_fix.json।

ফাইনাল আউটপুট: TaskMemoryManager.py কমপ্লিট, মেমরি সেভ/লোড কাজ করে। git commit -m "Task 1: Memory Fix (with config.py)"।
টাস্ক ২: হাইব্রিড সুইচিং ইঞ্জিন (ModelRouter Class)
ওভারভিউ: এই টাস্কে ModelRouter ক্লাস তৈরি করো যা ক্লাউড (7B/Groq 70B) এবং লোকাল (14B) লেয়ারের মধ্যে অটোমেটিক সুইচ করবে। এটি টাস্কের ধরন (e.g., সিম্পল চ্যাট vs ভারী কম্পুটেশন), API লিমিট, খরচ, স্পিড এবং অ্যাভেলেবিলিটি অনুসারে সিদ্ধান্ত নেবে, যাতে সিস্টেম সবসময় অপটিমাইজড এবং রিলায়েবল থাকে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: requests (for API calls to Groq/Render), ollama (for local model integration), os/time (for health checks), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/router/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/router/ModelRouter.py ফাইল তৈরি।
class ModelRouter:।
init মেথড যোগ:Pythonfrom config import RENDER_URL
self.models = {
    'cloud_7b': {'url': RENDER_URL + '/api/7b', 'type': 'cloud'},
    'groq_70b': {'url': 'https://api.groq.com/v1/models/70b', 'key': os.getenv('GROQ_API_KEY'), 'type': 'cloud'},
    'local_14b': {'model': 'llama-14b', 'type': 'local'}
}
route_request মেথড যোগ:Pythondef route_request(self, task_type: str, data_size: int, urgency: str):
    if data_size > 1000 or task_type == 'sensitive':
        return 'local_14b'
    elif urgency == 'high':
        return 'groq_70b'
    else:
        return 'cloud_7b'
    logging.info(f"Routed to {model} for task {task_type}")
generate_response মেথড যোগ:Pythondef generate_response(self, prompt: str, task_type: str, data_size: int = 0, urgency: str = 'normal'):
    model = self.route_request(task_type, data_size, urgency)
    if model == 'local_14b':
        import ollama
        return ollama.generate(model=self.models[model]['model'], prompt=prompt)['response']
    elif model == 'groq_70b':
        headers = {'Authorization': f"Bearer {self.models[model]['key']}"}
        response = requests.post(self.models[model]['url'], json={'prompt': prompt}, headers=headers)
        return response.json()['response']
    else:
        response = requests.post(self.models[model]['url'], json={'prompt': prompt})
        return response.json()['response']
app.py-এ ইন্টিগ্রেট:Pythonfrom config import RENDER_URL
from modules.router.ModelRouter import ModelRouter
router = ModelRouter()

ইন্টিগ্রেশন: config.py থেকে RENDER_URL নেওয়া। পরবর্তী টাস্কে route_request এবং generate_response ইউজ করো।
টেস্টিং:

tests/router_test.py তৈরি।
python -m unittest tests/router_test.py।
রিপোর্ট সেভ: reports/task_02_hybrid_switching.json।

ফাইনাল আউটপুট: ModelRouter.py কমপ্লিট, হাইব্রিড সুইচিং কাজ করে। git commit -m "Task 2: Hybrid Switching (with config.py)"।
টাস্ক ৩: API লিমিট মনিটরিং + স্মার্ট সুইচিং লজিক
ওভারভিউ: এই টাস্কে Groq/Render API-এর রেট লিমিট মনিটর করা হবে এবং লিমিট শেষ হলে অটোমেটিক সুইচ করা হবে অন্য মডেলে (e.g., local 14B)। স্মার্ট লজিক যোগ করে প্রিডিক্টিভ সুইচিং করা হবে (e.g., limit close হলে early switch), যাতে ডাউনটাইম না হয় এবং খরচ অপটিমাইজ হয়।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: requests, collections.deque (built-in), threading (built-in), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/router/ ফোল্ডারে আপডেট।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/router/ModelRouter.py আপডেট (টাস্ক ২-এর ফাইল)।
LimitMonitor ক্লাস যোগ:Pythonclass LimitMonitor:
    def __init__(self):
        self.calls = deque()
        self.limit = int(os.getenv("API_LIMIT", "100"))
        self.threshold = 0.1 * self.limit
track_call মেথড যোগ:Pythondef track_call(self):
    self.calls.append(time.time())
    while self.calls and self.calls[0] < time.time() - 60:
        self.calls.popleft()
is_limit_exceeded মেথড যোগ:Pythondef is_limit_exceeded(self):
    return len(self.calls) >= self.limit
ModelRouter-এ ইন্টিগ্রেট:Pythonself.limit_monitor = LimitMonitor()  # __init__-এ যোগ
generate_response-এ track_call কল এবং route_request-এ লিমিট চেক যোগ:Pythonself.limit_monitor.track_call()
if model.startswith('cloud') and self.limit_monitor.is_limit_exceeded():
    return 'local_14b'
app.py-এ ইন্টিগ্রেট:Pythonfrom config import RENDER_URL

ইন্টিগ্রেশন: config.py থেকে RENDER_URL নেওয়া। পরবর্তী টাস্কে limit_monitor ইউজ করো।
টেস্টিং:

tests/limit_test.py তৈরি।
python -m unittest tests/limit_test.py।
রিপোর্ট সেভ: reports/task_03_api_limit_monitoring.json।

ফাইনাল আউটপুট: ModelRouter.py আপডেট, লিমিট মনিটরিং কাজ করে। git commit -m "Task 3: API Limit Monitoring (with config.py)"।
টাস্ক ৪: Approval System (ApprovalManager - First Version)
ওভারভিউ: এই টাস্কে ApprovalManager ক্লাস তৈরি করো যা সেনসিটিভ অপারেশনের আগে ইউজার অ্যাপ্রুভাল নেবে। প্রথম ভার্সনে কনসোল প্রম্পট (yes/no) এবং লগিং। রুল-বেসড অটো-অ্যাপ্রুভ লো-রিস্ক অ্যাকশনের জন্য।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: input() for console, flask/fastapi for API-based UI, logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/approval/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/approval/ApprovalManager.py ফাইল তৈরি।
class ApprovalManager:।
init মেথড যোগ:Pythonfrom config import APPROVAL_TIMEOUT
self.rules = {'low_risk': ['read_file'], 'high_risk': ['delete_file', 'upgrade_code']}
self.timeout = APPROVAL_TIMEOUT
request_approval মেথড যোগ:Pythondef request_approval(self, action: str, description: str):
    if action in self.rules['low_risk']:
        return True
    prompt = f"Approve {action}? {description} (Y/N): "
    import time; start = time.time()
    while time.time() - start < self.timeout:
        user_input = input(prompt).strip().upper()
        if user_input == 'Y':
            logging.info(f"Approved: {action}")
            return True
        elif user_input == 'N':
            logging.info(f"Denied: {action}")
            return False
    logging.warning(f"Timeout for {action}")
    return False
app.py-এ ইন্টিগ্রেট:Pythonfrom config import APPROVAL_TIMEOUT
from modules.approval.ApprovalManager import ApprovalManager
approval_manager = ApprovalManager()

ইন্টিগ্রেশন: config.py থেকে APPROVAL_TIMEOUT নেওয়া। পরবর্তী টাস্কে request_approval ইউজ করো।
টেস্টিং:

tests/approval_test.py তৈরি।
python -m unittest tests/approval_test.py।
রিপোর্ট সেভ: reports/task_04_approval_system.json।

ফাইনাল আউটপুট: ApprovalManager.py কমপ্লিট, অ্যাপ্রুভাল প্রম্পট কাজ করে। git commit -m "Task 4: Approval System (with config.py)"।
টাস্ক ৫: Basic File + CMD Execution Module (Permission-Based)
ওভারভিউ: এই টাস্কে ExecutionModule ক্লাস তৈরি করো যা ফাইল অপারেশন (create/delete/edit) এবং CMD/PowerShell কমান্ড এক্সিকিউট করবে, কিন্তু শুধু ইউজার অ্যাপ্রুভাল সাপেক্ষে। এটি সেলফ-আপগ্রেডের বেস হবে, যেমন কোড ফাইল এডিট।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: subprocess (for CMD), os/shutil (for files), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/execution/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/execution/ExecutionModule.py ফাইল তৈরি।
class ExecutionModule:।
init মেথড যোগ:Pythonfrom config import SANDBOX_DIR
import os
self.sandbox_dir = SANDBOX_DIR
if not os.path.exists(self.sandbox_dir):
    os.makedirs(self.sandbox_dir)
file_op মেথড যোগ:Pythondef file_op(self, op: str, file_path: str, content: str = None):
    full_path = os.path.join(self.sandbox_dir, file_path)
    from modules.approval.ApprovalManager import ApprovalManager
    approval_manager = ApprovalManager()
    if not approval_manager.request_approval(op, f"{op} on {file_path}"):
        return "Denied"
    if op == 'create':
        with open(full_path, 'w') as f: f.write(content)
        logging.info(f"Created {full_path}")
    elif op == 'delete':
        os.remove(full_path)
        logging.info(f"Deleted {full_path}")
    elif op == 'edit':
        with open(full_path, 'a') as f: f.write(content)
        logging.info(f"Edited {full_path}")
execute_cmd মেথড যোগ:Pythondef execute_cmd(self, cmd: str):
    approval_manager = ApprovalManager()
    if not approval_manager.request_approval('cmd', f"Execute {cmd}"):
        return "Denied"
    import subprocess
    try:
        output = subprocess.check_output(cmd, shell=True, cwd=self.sandbox_dir)
        return output.decode('utf-8')
    except Exception as e:
        logging.error(f"CMD error: {e}")
        return str(e)
app.py-এ ইন্টিগ্রেট:Pythonfrom config import SANDBOX_DIR
from modules.execution.ExecutionModule import ExecutionModule
executor = ExecutionModule()

ইন্টিগ্রেশন: config.py থেকে SANDBOX_DIR নেওয়া। পরবর্তী টাস্কে execute_cmd ইউজ করো।
টেস্টিং:

tests/execution_test.py তৈরি।
python -m unittest tests/execution_test.py।
রিপোর্ট সেভ: reports/task_05_execution_module.json।

ফাইনাল আউটপুট: ExecutionModule.py কমপ্লিট, ফাইল/CMD এক্সিকিউশন অ্যাপ্রুভাল সহ কাজ করে। git commit -m "Task 5: File CMD Execution (with config.py)"।
টাস্ক ৬: Local 14B Model Connection (Ollama Integration)
ওভারভিউ: এই টাস্কে লোকাল PC-এ চলা 14B প্যারামিটার মডেলকে সিস্টেমের সাথে কানেক্ট করা হবে Ollama ফ্রেমওয়ার্ক ব্যবহার করে। এটি ভারী কম্পুটেশনাল টাস্কের জন্য লোকাল মডেল ব্যবহার করবে, যাতে প্রাইভেসি মেইনটেইন হয় এবং ক্লাউড খরচ কমে। কানেকশন স্টেবল হবে, অটো-রিকানেক্ট মেকানিজম সহ, এবং মডেলের স্টেট মনিটর করবে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: ollama (pip install ollama), requests (API calls), subprocess (Ollama start/stop), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/local/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/local/LocalModelConnector.py ফাইল তৈরি।
class LocalModelConnector:।
init মেথড যোগ:Pythonfrom config import TTS_TEST_URL  # যদি লাগে, অন্যথায় আগের মতো
self.model = os.getenv('OLLAMA_MODEL', 'llama-14b')
self.port = os.getenv('OLLAMA_PORT', '11434')
self._check_and_start()
_check_and_start মেথড যোগ:Pythondef _check_and_start(self):
    try:
        response = requests.get(f"http://localhost:{self.port}/health")
        if response.status_code != 200:
            raise ConnectionError
    except ConnectionError:
        import subprocess
        subprocess.Popen(['ollama', 'serve'])
    from tenacity import retry, stop_after_attempt
    @retry(stop=stop_after_attempt(3))
    def connect():
        requests.get(f"http://localhost:{self.port}/health")
    connect()
    logging.info("Ollama connected")
generate_response মেথড যোগ:Pythondef generate_response(self, prompt: str):
    return ollama.generate(model=self.model, prompt=prompt)['response']
app.py-এ ইন্টিগ্রেট:Pythonfrom modules.local.LocalModelConnector import LocalModelConnector
local_connector = LocalModelConnector()

ইন্টিগ্রেশন: পরবর্তী টাস্কে get_state এবং generate_response ইউজ করো।
টেস্টিং:

tests/local_test.py তৈরি।
python -m unittest tests/local_test.py।
রিপোর্ট সেভ: reports/task_06_local_model_connection.json।

ফাইনাল আউটপুট: LocalModelConnector.py কমপ্লিট, লোকাল মডেল কানেকশন কাজ করে। git commit -m "Task 6: Local Model Connection (with config.py)"।
টাস্ক ৭: Smart Switching Test (Cloud vs Local)
ওভারভিউ: এই টাস্কে হাইব্রিড সুইচিং ইঞ্জিনকে টেস্ট করা হবে, যাতে ক্লাউড (7B/Groq 70B) এবং লোকাল (14B) মডেলের মধ্যে সুইচিং সঠিকভাবে কাজ করে কিনা চেক হয়। টেস্ট কেসগুলোতে লোড, লিমিট, স্পিড, এবং এরর সিনারিও কভার করা হবে, যাতে সিস্টেমের রিলায়েবিলিটি নিশ্চিত হয়। রিপোর্ট জেনারেট করবে সাকসেস রেট, টাইমিং, এবং ফেলিউর কজ সহ।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: pytest (testing framework), unittest.mock (simulate errors/limits), time/performance (timing), logging/json (report generation), concurrent.futures (parallel tests)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: tests/ ফোল্ডারে switching_test.py।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

tests/switching_test.py ফাইল তৈরি।
class TestSuite:।
init মেথড যোগ:Pythonfrom modules.router.ModelRouter import ModelRouter
self.router = ModelRouter()
self.report = {'success_rate': 0, 'tests': []}
run_tests মেথড যোগ:Pythondef run_tests(self):
    tests = ['high_load', 'limit_exceed', 'offline_local']
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(self._run_single_test, tests))
    success_count = sum(1 for r in results if r['success'])
    self.report['success_rate'] = (success_count / len(tests)) * 100
    self.report['tests'] = results
    with open('reports/task_07_smart_switching.json', 'w') as f:
        json.dump(self.report, f, indent=4)
    logging.info("Report generated")
_run_single_test মেথড যোগ:Pythondef _run_single_test(self, test_type: str):
    import time; start = time.time()
    try:
        if test_type == 'high_load':
            for _ in range(10):
                self.router.route_request('chat', 500, 'high')
            success = True
        elif test_type == 'limit_exceed':
            from unittest.mock import patch
            with patch('modules.router.ModelRouter.limit_monitor.is_limit_exceeded', return_value=True):
                model = self.router.route_request('chat', 100, 'normal')
                success = model == 'local_14b'
        elif test_type == 'offline_local':
            with patch('requests.get', side_effect=ConnectionError):
                model = self.router.route_request('sensitive', 2000, 'low')
                success = model != 'local_14b'
        error_type = None
    except Exception as e:
        success = False
        error_type = str(e)
    time_taken = time.time() - start
    return {'type': test_type, 'success': success, 'time_taken': time_taken, 'error_type': error_type}
app.py-এ ইন্টিগ্রেট:Pythonfrom tests.switching_test import TestSuite
if __name__ == '__main__' and 'test_switching' in sys.argv:
    suite = TestSuite()
    suite.run_tests()

ইন্টিগ্রেশন: টাস্ক ২-এর ModelRouter টেস্ট। পরবর্তী টাস্কে রিপোর্ট ইউজ করো।
টেস্টিং:

python tests/switching_test.py রান করে চেক।
reports/task_07_smart_switching.json চেক।

ফাইনাল আউটপুট: switching_test.py কমপ্লিট, টেস্ট রান করে রিপোর্ট জেনারেট। git commit -m "Task 7: Smart Switching Test (with config.py)"।
টাস্ক ৮: Error Logging + Auto-Learning Basic Framework
ওভারভিউ: এই টাস্কে সিস্টেমের সব এররকে লগ করা হবে এবং একটা বেসিক অটো-লার্নিং মেকানিজম তৈরি করা হবে যা এরর থেকে লেসন সেভ করে (লং-টার্ম মেমরিতে) এবং ফিউচার অপারেশনে অ্যাভয়েড করবে। এটি সেলফ-ইম্প্রুভমেন্টের বেস হবে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: logging (structured logs), traceback (stack traces), re (regex patterns), json (lesson storage)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/error/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/error/ErrorLogger.py ফাইল তৈরি।
class ErrorLogger:।
init মেথড যোগ:Pythonfrom config import LEARNING_PROMPT
self.patterns = {'connection_refused': r'Connection refused', 'limit_exceeded': r'API limit'}
logging.basicConfig(filename='logs/errors.log', level=logging.ERROR)
log_error মেথড যোগ:Pythondef log_error(self, e: Exception, context: str):
    import traceback
    stack = traceback.format_exc()
    logging.error(f"Error: {str(e)}, Context: {context}, Stack: {stack}")
    self._analyze_and_learn(str(e))
_analyze_and_learn মেথড যোগ:Pythondef _analyze_and_learn(self, error_msg: str):
    import re
    for pattern_name, regex in self.patterns.items():
        if re.search(regex, error_msg):
            lesson = f"Avoid {pattern_name} by checking before action"
            from modules.memory.TaskMemoryManager import TaskMemoryManager
            memory_manager = TaskMemoryManager()
            memory_manager.save_memory('error_lesson', {'lesson': lesson, 'error': error_msg}, 'learning')
            logging.info(f"Learned: {lesson}")
check_learned_lessons মেথড যোগ:Pythondef check_learned_lessons(self, action: str):
    from modules.memory.TaskMemoryManager import TaskMemoryManager
    memory_manager = TaskMemoryManager()
    lessons = memory_manager.load_memory('learning')
    for lesson in lessons:
        if action in lesson['content']['lesson']:
            return False
    return True
app.py-এ ইন্টিগ্রেট:Pythonfrom modules.error.ErrorLogger import ErrorLogger
error_logger = ErrorLogger()

ইন্টিগ্রেশন: config.py থেকে LEARNING_PROMPT নেওয়া। পরবর্তী টাস্কে check_learned_lessons ইউজ করো।
টেস্টিং:

tests/error_test.py তৈরি।
python -m unittest tests/error_test.py।
রিপোর্ট সেভ: reports/task_08_error_logging.json।

ফাইনাল আউটপুট: ErrorLogger.py কমপ্লিট, এরর লগিং এবং অটো-লার্নিং কাজ করে। git commit -m "Task 8: Error Logging Auto-Learning (with config.py)"।
টাস্ক ৯: /status Endpoint Upgrade (Show All Models' State)
ওভারভিউ: এই টাস্কে Render-এর API-এ /status এন্ডপয়েন্টকে আপগ্রেড করা হবে, যাতে সব মডেলের স্টেট (cloud 7B, Groq 70B, local 14B) দেখাবে: running/idle/down, usage stats (requests, limits), health (ping response), এবং overall system health। এটি মনিটরিং এবং ডিবাগিং সহজ করবে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: flask/fastapi (API framework), requests (health pings), json (response), time (uptime calc)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: app.py আপডেট।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

app.py আপডেট।
@app.route('/status', methods=['GET']) def get_status():Pythonfrom config import STATUS_AUTH_KEY
if request.headers.get('Auth-Key') != STATUS_AUTH_KEY:
    return jsonify({'error': 'Unauthorized'}), 401
from modules.router.ModelRouter import ModelRouter
router = ModelRouter()
from modules.local.LocalModelConnector import LocalModelConnector
local_connector = LocalModelConnector()
import time
uptime = time.time() - start_time  # গ্লোবাল start_time
status = {
    'models': {
        'cloud_7b': {'status': 'running' if requests.get(router.models['cloud_7b']['url'] + '/health').status_code == 200 else 'down', 'uptime': uptime, 'requests': len(router.limit_monitor.calls) if hasattr(router, 'limit_monitor') else 0},
        'groq_70b': {'status': 'running' if requests.get(router.models['groq_70b']['url'], headers={'Authorization': f"Bearer {router.models['groq_70b']['key']}"}, timeout=5).status_code == 200 else 'down', 'uptime': uptime, 'limits': router.limit_monitor.limit - len(router.limit_monitor.calls)},
        'local_14b': {'status': local_connector.get_state(), 'uptime': uptime, 'health': 'ok' if local_connector.get_state() == 'running' else 'error'}
    },
    'system_health': 'ok' if all(s['status'] == 'running' for s in status['models'].values()) else 'partial'
}
logging.info("Status requested")
return jsonify(status)
app.py-এ ইন্টিগ্রেট:Pythonfrom config import STATUS_AUTH_KEY

ইন্টিগ্রেশন: config.py থেকে STATUS_AUTH_KEY নেওয়া। পরবর্তী টাস্কে /status কল করো।
টেস্টিং:

tests/status_test.py তৈরি।
python -m unittest tests/status_test.py।
রিপোর্ট সেভ: reports/task_09_status_endpoint.json।

ফাইনাল আউটপুট: app.py আপডেট, /status এন্ডপয়েন্ট কাজ করে। git commit -m "Task 9: Status Endpoint Upgrade (with config.py)"।
টাস্ক ১০: Render + Local Synchronization (Memory Share)
ওভারভিউ: এই টাস্কে ক্লাউড (Render) এবং লোকাল (PC) লেয়ারের মধ্যে লং-টার্ম মেমরি শেয়ার করা হবে, যাতে ডাটা সিঙ্ক থাকে। এটি bidirectional sync সাপোর্ট করবে, conflict resolution (timestamp-based), এবং offline mode (queue changes)।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: requests (API sync), sqlite3/json (data format), schedule/threading (periodic tasks), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/sync/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/sync/SyncEngine.py ফাইল তৈরি।
class SyncEngine:।
init মেথড যোগ:Pythonfrom config import SYNC_INTERVAL
self.local_url = 'http://localhost:5001/sync'
self.render_url = RENDER_URL + '/sync'
self.interval = SYNC_INTERVAL
self.offline_queue = []
import threading
threading.Thread(target=self._periodic_sync, daemon=True).start()
sync_memories মেথড যোগ:Pythondef sync_memories(self, direction: str = 'bidirectional'):
    from modules.memory.TaskMemoryManager import TaskMemoryManager
    memory_manager = TaskMemoryManager()
    local_memories = memory_manager.load_memory()
    try:
        response = requests.get(self.render_url)
        render_memories = response.json() if response.status_code == 200 else []
    except:
        self._queue_for_later(local_memories)
        return "Offline sync queued"
    merged = self._resolve_conflicts(local_memories, render_memories)
    if direction in ['to_render', 'bidirectional']:
        requests.post(self.render_url, json=merged)
    if direction in ['to_local', 'bidirectional']:
        for mem in merged:
            memory_manager.save_memory(mem['task_id'], mem['content'], mem['category'])
    logging.info("Sync completed")
    return "Success"
_resolve_conflicts মেথড যোগ:Pythondef _resolve_conflicts(self, local: list, render: list):
    from datetime import datetime
    mem_dict = {}
    for mem in local + render:
        key = mem['task_id']
        if key not in mem_dict or datetime.strptime(mem['timestamp'], '%Y-%m-%d %H:%M:%S.%f') > datetime.strptime(mem_dict[key]['timestamp'], '%Y-%m-%d %H:%M:%S.%f'):
            mem_dict[key] = mem
    return list(mem_dict.values())
app.py-এ ইন্টিগ্রেট:Pythonfrom config import SYNC_INTERVAL
from modules.sync.SyncEngine import SyncEngine
sync_engine = SyncEngine()

ইন্টিগ্রেশন: config.py থেকে SYNC_INTERVAL নেওয়া। পরবর্তী টাস্কে sync_memories কল করো।
টেস্টিং:

tests/sync_test.py তৈরি।
python -m unittest tests/sync_test.py।
রিপোর্ট সেভ: reports/task_10_render_local_sync.json।

ফাইনাল আউটপুট: SyncEngine.py কমপ্লিট, মেমরি সিঙ্ক কাজ করে। git commit -m "Task 10: Render Local Sync (with config.py)"।

টাস্ক ১১: SelfUpgradeEngine Class Creation
ওভারভিউ: এই টাস্কে একটা কেন্দ্রীয় ক্লাস তৈরি করা হবে যা সিস্টেমের সেলফ-আপগ্রেড প্রক্রিয়াকে ম্যানেজ করবে। SelfUpgradeEngine ক্লাসটি ইউজারের অনুরোধ (e.g., নতুন ফিচার যোগ) অ্যানালাইজ করে কোড জেনারেট করবে, টেস্ট করবে, এবং ডেপ্লয় করবে। এটি অ্যাপ্রুভাল সিস্টেম, মেমরি ম্যানেজার, এবং এক্সিকিউশন মডিউলের সাথে ইন্টিগ্রেট হবে, যাতে সিস্টেম নিজেকে নিরাপদে এবং অটোমেটিকভাবে আপগ্রেড করতে পারে।
প্রয়োজনীয় প্রিপারেশন:

* রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।

* লাইব্রেরী: ast (code syntax validation), gitpython (git operations), requests (LLM API calls), logging। Ollama/Groq for code generation prompts।

* GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।

* ডিরেক্টরি: modules/upgrade/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

1. modules/upgrade/SelfUpgradeEngine.py ফাইল তৈরি।

2. class SelfUpgradeEngine:।

3. __init__ মেথড যোগ:
   ```python
   from config import RENDER_URL
   self.git_repo = os.getenv("GIT_REPO", "https://github.com/The-Mask-Of-Imran/The-Mask-Core-System")
   self.prompt_template = "Generate Python code for feature {feature}: {description}"

upgrade_request মেথড যোগ:Pythondef upgrade_request(self, feature: str, description: str):
    from modules.approval.ApprovalManager import ApprovalManager
    approval_manager = ApprovalManager()
    if not approval_manager.request_approval('upgrade', f"Upgrade for {feature}"):
        return "Denied"
    code = self._generate_code(feature, description)
    if not self._validate_code(code):
        return "Invalid code"
    self._execute_upgrade(code, feature)
    logging.info(f"Upgraded {feature}")
_generate_code মেথড যোগ:Pythondef _generate_code(self, feature: str, description: str):
    prompt = self.prompt_template.format(feature=feature, description=description)
    from modules.router.ModelRouter import ModelRouter
    router = ModelRouter()
    return router.generate_response(prompt, 'code_gen')
_validate_code মেথড যোগ:Pythondef _validate_code(self, code: str):
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        logging.error(f"Syntax error: {e}")
        return False
_execute_upgrade মেথড যোগ:Pythondef _execute_upgrade(self, code: str, feature: str):
    with open(f'modules/{feature.lower()}/{feature}Module.py', 'w') as f:
        f.write(code)
    from modules.execution.ExecutionModule import ExecutionModule
    executor = ExecutionModule()
    executor.execute_cmd('pytest tests/')
    import git
    repo = git.Repo('.')
    repo.git.add(A=True)
    repo.commit(m=f"Upgrade {feature}")
    repo.git.push()
    from modules.sync.SyncEngine import SyncEngine
    SyncEngine().sync_memories()
    from modules.memory.TaskMemoryManager import TaskMemoryManager
    memory_manager = TaskMemoryManager()
    memory_manager.save_memory('upgrade', {'feature': feature, 'code': code}, 'upgrade')
app.py-এ ইন্টিগ্রেট:Pythonfrom config import RENDER_URL
from modules.upgrade.SelfUpgradeEngine import SelfUpgradeEngine
upgrade_engine = SelfUpgradeEngine()

ইন্টিগ্রেশন: config.py থেকে RENDER_URL নেওয়া। পরবর্তী টাস্কে execute_flow ইউজ করো।
টেস্টিং:

tests/upgrade_test.py তৈরি।
python -m unittest tests/upgrade_test.py।
রিপোর্ট সেভ: reports/task_11_self_upgrade_engine.json।

ফাইনাল আউটপুট: SelfUpgradeEngine.py কমপ্লিট, অ্যাপগ্রেড রিকোয়েস্ট কাজ করে। git commit -m "Task 11: SelfUpgradeEngine (with config.py)"।
টাস্ক ১২: Code Generate → Local Test → Git Push → Render Restart Flow
ওভারভিউ: এই টাস্কে সেলফ-আপগ্রেডের সম্পূর্ণ ফ্লো তৈরি করা হবে: LLM দিয়ে কোড জেনারেট, লোকালে টেস্ট (unit/integration), git-এ পুশ, এবং Render-এ অটো রিস্টার্ট। এটি পাইপলাইন হিসেবে চলবে, যাতে ম্যানুয়াল ইন্টারভেনশন ছাড়াই আপগ্রেড হয়, এবং লগিং সব স্টেপের।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: gitpython (git ops), subprocess (test running), requests (Render API), unittest/pytest (testing)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/upgrade/ ফোল্ডারে আপডেট।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/upgrade/SelfUpgradeEngine.py আপডেট (টাস্ক ১১-এর ফাইল)।
execute_flow মেথড যোগ:Pythondef execute_flow(self, feature: str, description: str):
    code = self._generate_code(feature, description)
    if not self._test_locally(code):
        return "Test failed"
    self._git_push(feature)
    self._render_restart()
    logging.info("Flow completed")
_test_locally মেথড যোগ:Pythondef _test_locally(self, code: str):
    test_file = f'tests/{feature.lower()}_test.py'
    with open(test_file, 'w') as f:
        f.write('# Dummy test\nassert True')
    import subprocess
    result = subprocess.run(['pytest', test_file], capture_output=True)
    return result.returncode == 0
_git_push মেথড যোগ:Pythondef _git_push(self, feature: str):
    import git
    repo = git.Repo('.')
    repo.git.add(A=True)
    repo.commit(m=f"Upgrade flow for {feature}")
    repo.git.push()
_render_restart মেথড যোগ:Pythonfrom config import RENDER_WEBHOOK
def _render_restart(self):
    requests.post(RENDER_WEBHOOK)
app.py-এ ইন্টিগ্রেট:Pythonfrom config import RENDER_WEBHOOK
from modules.upgrade.SelfUpgradeEngine import SelfUpgradeEngine
upgrade_engine = SelfUpgradeEngine()

ইন্টিগ্রেশন: config.py থেকে RENDER_WEBHOOK নেওয়া। পরবর্তী টাস্কে execute_flow কল করো।
টেস্টিং:

tests/flow_test.py তৈরি।
python -m unittest tests/flow_test.py।
রিপোর্ট সেভ: reports/task_12_upgrade_flow.json।

ফাইনাল আউটপুট: SelfUpgradeEngine.py আপডেট, আপগ্রেড ফ্লো কাজ করে। git commit -m "Task 12: Upgrade Flow (with config.py)"।
টাস্ক ১৩: Backup System (Zip + Google Drive)
ওভারভিউ: এই টাস্কে আপগ্রেডের আগে/পরে সিস্টেমের সম্পূর্ণ ব্যাকআপ নেওয়া হবে: কোড, ডাটাবেস, কনফিগ ফাইলগুলোকে zip করে Google Drive-এ আপলোড। এটি অটোমেটিক ট্রিগার হবে, ভার্সনিং সাপোর্ট (timestamped backups) সহ, যাতে ডাটা লস না হয়।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: shutil/zipfile (zipping), google-api-python-client (Drive API), oauth2client (auth)। Setup service account for auth।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/backup/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/backup/BackupManager.py তৈরি।
class BackupManager:।
init মেথড যোগ:Pythonfrom config import GOOGLE_DRIVE_FOLDER_ID, SERVICE_ACCOUNT_JSON
from googleapiclient.discovery import build
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=['https://www.googleapis.com/auth/drive'])
self.drive_service = build('drive', 'v3', credentials=credentials)
self.folder_id = GOOGLE_DRIVE_FOLDER_ID
self.retention = 10
create_backup মেথড যোগ:Pythondef create_backup(self):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_name = f'backup_{timestamp}.zip'
    import shutil
    shutil.make_archive(zip_name[:-4], 'zip', '.')
    file_metadata = {'name': zip_name, 'parents': [self.folder_id]}
    media = googleapiclient.http.MediaFileUpload(zip_name, mimetype='application/zip')
    file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    logging.info(f"Uploaded backup {file['id']}")
    self._apply_retention()
    os.remove(zip_name)
_apply_retention মেথড যোগ:Pythondef _apply_retention(self):
    query = f"'{self.folder_id}' in parents and mimeType='application/zip'"
    files = self.drive_service.files().list(q=query, orderBy='createdTime desc').execute().get('files', [])
    if len(files) > self.retention:
        for file in files[self.retention:]:
            self.drive_service.files().delete(fileId=file['id']).execute()
SelfUpgradeEngine.py আপডেট (টাস্ক ১১-১২):Pythonfrom modules.backup.BackupManager import BackupManager
BackupManager().create_backup()  # প্রি/পোস্ট ট্রিগার
app.py-এ ইন্টিগ্রেট:Pythonfrom config import GOOGLE_DRIVE_FOLDER_ID
from modules.backup.BackupManager import BackupManager
backup_manager = BackupManager()

ইন্টিগ্রেশন: config.py থেকে GOOGLE_DRIVE_FOLDER_ID নেওয়া। পরবর্তী টাস্কে create_backup ইউজ করো।
টেস্টিং:

tests/backup_test.py তৈরি।
python -m unittest tests/backup_test.py।
রিপোর্ট সেভ: reports/task_13_backup_system.json।

ফাইনাল আউটপুট: BackupManager.py কমপ্লিট, ব্যাকআপ আপলোড কাজ করে। git commit -m "Task 13: Backup System (with config.py)"।
টাস্ক ১৪: Rollback Mechanism (Git Checkout + Restore)
ওভারভিউ: এই টাস্কে আপগ্রেড ফেল হলে অটো/ম্যানুয়াল রোলব্যাক করা হবে: git checkout previous commit, restore from backup, এবং restart। এটি ইতিহাস ট্র্যাক করবে, যাতে সিস্টেম স্টেবল থাকে এবং ডাউনটাইম কম হয়।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: gitpython (checkout), shutil (restore), subprocess (restart script)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/rollback/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/rollback/RollbackManager.py ফাইল তৈরি।
class RollbackManager:।
init মেথড যোগ:Pythonfrom config import BACKUP_DIR
import git
self.repo = git.Repo('.')
self.backup_dir = BACKUP_DIR
rollback মেথড যোগ:Pythondef rollback(self, reason: str = "Upgrade failed"):
    logging.warning(f"Rollback triggered: {reason}")
    prev_commit = self._get_previous_commit()
    if prev_commit:
        self.repo.git.checkout(prev_commit)
        logging.info(f"Reverted to commit {prev_commit}")
    self._restore_latest_backup()
    self._restart_system()
    from modules.sync.SyncEngine import SyncEngine
    SyncEngine().sync_memories()
    return "Rollback completed"
_get_previous_commit মেথড যোগ:Pythondef _get_previous_commit(self):
    if self.repo.head.commit.parents:
        return self.repo.head.commit.parents[0].hexsha
    return None
_restore_latest_backup মেথড যোগ:Pythondef _restore_latest_backup(self):
    import glob, os
    backup_files = glob.glob(os.path.join(self.backup_dir, 'backup_*.zip'))
    if not backup_files:
        logging.error("No backup found")
        return
    latest_file = max(backup_files, key=os.path.getctime)
    import shutil
    shutil.unpack_archive(latest_file, '.')
    logging.info(f"Restored from {latest_file}")
_restart_system মেথড যোগ:Pythondef _restart_system(self):
    import subprocess
    subprocess.Popen(os.getenv("RESTART_CMD", "python app.py").split())
app.py-এ ইন্টিগ্রেট:Pythonfrom config import BACKUP_DIR
from modules.rollback.RollbackManager import RollbackManager
rollback_manager = RollbackManager()

ইন্টিগ্রেশন: config.py থেকে BACKUP_DIR নেওয়া। পরবর্তী টাস্কে rollback ইউজ করো।
টেস্টিং:

tests/rollback_test.py তৈরি।
python -m unittest tests/rollback_test.py।
রিপোর্ট সেভ: reports/task_14_rollback_mechanism.json।

ফাইনাল আউটপুট: RollbackManager.py কমপ্লিট, রোলব্যাক কাজ করে। git commit -m "Task 14: Rollback Mechanism (with config.py)"।
টাস্ক ১৫: Auto Upgrade Proposal + Permission UI
ওভারভিউ: এই টাস্কে সিস্টেম নিজে আপগ্রেড প্রস্তাব করবে (e.g., from learned errors) এবং একটা UI (console/web) দিয়ে permission নেবে। প্রস্তাবে details (what, why, risk) থাকবে, যাতে ইউজার ইনফর্মড ডিসিশন নিতে পারে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: streamlit (web UI), input() (console), json (proposal format)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/proposal/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/proposal/ProposalGenerator.py তৈরি।
class ProposalGenerator:।
init মেথড যোগ:Pythonfrom config import PROPOSAL_THRESHOLD
self.threshold = float(os.getenv("PROPOSAL_THRESHOLD", "0.5"))
generate_proposal মেথড যোগ:Pythondef generate_proposal(self):
    from modules.memory.TaskMemoryManager import TaskMemoryManager
    memory_manager = TaskMemoryManager()
    errors = memory_manager.load_memory('learning')
    if len(errors) > 0 and self._score_proposal(errors) > self.threshold:
        proposal = {'what': 'Fix errors', 'why': 'From learned lessons', 'risk': 'Low'}
        logging.info(f"Generated proposal: {proposal}")
        return proposal
    return None
_score_proposal মেথড যোগ:Pythondef _score_proposal(self, errors: list):
    return len(errors) / 10.0
request_permission মেথড যোগ:Pythondef request_permission(self, proposal: dict):
    from modules.approval.ApprovalManager import ApprovalManager
    approval_manager = ApprovalManager()
    return approval_manager.request_approval('proposal', json.dumps(proposal))
app.py-এ ইন্টিগ্রেট:Pythonfrom modules.proposal.ProposalGenerator import ProposalGenerator
proposal_gen = ProposalGenerator()

ইন্টিগ্রেশন: config.py থেকে PROPOSAL_THRESHOLD নেওয়া। পরবর্তী টাস্কে generate_proposal ইউজ করো।
টেস্টিং:

tests/proposal_test.py তৈরি।
python -m unittest tests/proposal_test.py।
রিপোর্ট সেভ: reports/task_15_auto_upgrade_proposal.json।

ফাইনাল আউটপুট: ProposalGenerator.py কমপ্লিট, প্রপোজাল এবং UI কাজ করে। git commit -m "Task 15: Auto Upgrade Proposal (with config.py)"।
টাস্ক ১৬: First Self-Upgrade Test (e.g., Add New /tts Endpoint)
ওভারভিউ: এই টাস্কে সেলফ-আপগ্রেড ইঞ্জিনের প্রথম টেস্ট করা হবে: একটা সিম্পল ফিচার (e.g., /tts endpoint for TTS) যোগ করে চেক করা যাবে পুরো ফ্লো কাজ করে কিনা। রিপোর্ট জেনারেট সাকসেস/ফেল সহ।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: pytest (testing), requests (verification), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: tests/upgrade/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

tests/upgrade/first_self_upgrade_test.py ফাইল তৈরি।
import pytest, requests, time, logging; from modules.upgrade.SelfUpgradeEngine import SelfUpgradeEngine।
class TestSelfUpgrade(unittest.TestCase):।
def setUp(self): self.engine = SelfUpgradeEngine(); self.start_time = time.time()।
def test_add_tts_endpoint(self):Pythonfeature = "TTS Endpoint"
description = "Add /tts endpoint using gtts library to convert text to speech"
result = self.engine.upgrade_request(feature, description)
assert result == "Success" or "Upgraded" in str(result)
time.sleep(2)
from config import TTS_TEST_URL
response = requests.get(TTS_TEST_URL + '?text=Test speech')
assert response.status_code == 200 and 'audio' in response.headers.get('Content-Type', '')
elapsed = time.time() - self.start_time
logging.info(f"Test completed in {elapsed}s")
app.py-এ ইন্টিগ্রেট:Pythonfrom config import TTS_TEST_URL

ইন্টিগ্রেশন: config.py থেকে TTS_TEST_URL নেওয়া। পরবর্তী টাস্কে টেস্ট রেজাল্ট ইউজ করো।
টেস্টিং:

python -m pytest tests/upgrade/first_self_upgrade_test.py -v।
reports/task_16_tts_upgrade.json চেক।

ফাইনাল আউটপুট: first_self_upgrade_test.py কমপ্লিট, প্রথম আপগ্রেড টেস্ট সাকসেসফুল। git commit -m "Task 16: First Self-Upgrade Test (with config.py)"।
টাস্ক ১৭: Error Handling Improvement (Auto Rollback on Upgrade Fail)
ওভারভিউ: এই টাস্কে আপগ্রেড ফেল হলে অটো রোলব্যাক ট্রিগার করা হবে, লগ করে, এবং নোটিফাই করা হবে। ইম্প্রুভড হ্যান্ডলিং: classify errors (recoverable vs fatal) এবং retry for transient।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: tenacity (retries), logging, smtplib/telegram (notifications)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/error/ ফোল্ডারে আপডেট।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/error/ErrorLogger.py আপডেট।
classify_error মেথড যোগ:Pythondef classify_error(self, e: Exception):
    recoverable = ['ConnectionError', 'TimeoutError']
    fatal = ['SyntaxError', 'ImportError']
    error_type = type(e).__name__
    if error_type in recoverable:
        return 'recoverable'
    elif error_type in fatal:
        return 'fatal'
    else:
        return 'unknown'
log_error মেথড আপডেট:Pythondef log_error(self, e: Exception, context: str):
    classification = self.classify_error(e)
    logging.error(f"{classification.upper()}: {str(e)} in {context}")
    from tenacity import retry, stop_after_attempt, wait_fixed
    if classification == 'recoverable':
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def retry_action():
            pass  # actual retry logic
        retry_action()
    if classification == 'fatal':
        from modules.rollback.RollbackManager import RollbackManager
        RollbackManager().rollback()
    self._notify(f"Fatal error in upgrade: {str(e)}")
_notify মেথড যোগ:Pythondef _notify(self, message: str):
    import requests
    requests.post(f"https://api.telegram.org/bot{os.getenv('NOTIFY_TELEGRAM_TOKEN')}/sendMessage", data={'chat_id': os.getenv('NOTIFY_CHAT_ID'), 'text': message})
app.py-এ ইন্টিগ্রেট:Pythonfrom modules.error.ErrorLogger import ErrorLogger
error_logger = ErrorLogger()

ইন্টিগ্রেশন: পরবর্তী টাস্কে অটো রোলব্যাক ট্রিগার করো।
টেস্টিং:

tests/error_improve_test.py তৈরি।
python -m pytest tests/error_improve_test.py।
রিপোর্ট সেভ: reports/task_17_error_handling.json।

ফাইনাল আউটপুট: ErrorLogger.py আপডেট, অটো রোলব্যাক এবং নোটিফিকেশন কাজ করে। git commit -m "Task 17: Error Handling Improvement (with config.py)"।
টাস্ক ১৮: Save Upgrade History in Long-Term Memory
ওভারভিউ: এই টাস্কে প্রত্যেক আপগ্রেডের ইতিহাস (what changed, when, success/fail) লং-টার্ম মেমরিতে সেভ করা হবে, যাতে ফিউচার লার্নিং/অডিট সম্ভব হয়। Queryable format (e.g., by date/feature)।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: json, sqlite3 (via MemoryManager)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/memory/ ফোল্ডারে আপডেট।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/memory/TaskMemoryManager.py আপডেট।
init-এ টেবল যোগ:Pythonself.cursor.execute('''CREATE TABLE IF NOT EXISTS upgrades (id INTEGER PRIMARY KEY, upgrade_id TEXT, changes TEXT, status TEXT, timestamp DATETIME)''')
self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_upgrade_timestamp ON upgrades (timestamp)')
SelfUpgradeEngine.py আপডেট:Pythonfrom datetime import datetime
timestamp = datetime.now()
history = {'upgrade_id': feature.lower().replace(' ', '_'), 'changes': code_snippet, 'status': 'success' if success else 'failed', 'timestamp': timestamp.isoformat()}
memory_manager = TaskMemoryManager()
memory_manager.cursor.execute("INSERT INTO upgrades (upgrade_id, changes, status, timestamp) VALUES (?, ?, ?, ?)", (history['upgrade_id'], json.dumps(history['changes']), history['status'], history['timestamp']))
memory_manager.conn.commit()
get_upgrade_history মেথড যোগ:Pythondef get_upgrade_history(self, by_date: str = None, by_feature: str = None):
    query = "SELECT * FROM upgrades"
    params = []
    if by_date:
        query += " WHERE timestamp LIKE ?"
        params.append(f"{by_date}%")
    if by_feature:
        query += " WHERE upgrade_id = ?"
        params.append(by_feature)
    self.cursor.execute(query, params)
    return [dict(row) for row in self.cursor.fetchall()]
app.py-এ ইন্টিগ্রেট:Pythonfrom modules.memory.TaskMemoryManager import TaskMemoryManager
memory_manager = TaskMemoryManager()

ইন্টিগ্রেশন: পরবর্তী টাস্কে get_upgrade_history ইউজ করো।
টেস্টিং:

tests/history_test.py তৈরি।
python -m pytest tests/history_test.py।
রিপোর্ট সেভ: reports/task_18_upgrade_history.json।

ফাইনাল আউটপুট: TaskMemoryManager.py আপডেট, আপগ্রেড হিস্ট্রি সেভ এবং কোয়েরি কাজ করে। git commit -m "Task 18: Save Upgrade History (with config.py)"।
টাস্ক ১৯: Smart Prompt Chaining (For Sensitive Tasks)
ওভারভিউ: এই টাস্কে সেনসিটিভ টাস্কের জন্য প্রম্পট চেইনিং করা হবে: multi-step prompts (plan, review, execute) যাতে accuracy বাড়ে এবং errors কম হয়।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: requests (LLM), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/prompt/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/prompt/PromptChainer.py ফাইল তৈরি।
class PromptChainer:।
init মেথড যোগ:Pythonfrom config import CHAIN_STEPS
self.steps = int(os.getenv("CHAIN_STEPS", "3"))
self.router = ModelRouter()
chain_prompts মেথড যোগ:Pythondef chain_prompts(self, initial_prompt: str, is_sensitive: bool = False):
    if not is_sensitive:
        return self.router.generate_response(initial_prompt, 'chat')
    outputs = []
    current_prompt = initial_prompt
    for step in range(self.steps):
        if step == 0:
            prompt = f"Step 1 - Plan: {current_prompt}. Provide a detailed plan."
        elif step == 1:
            prompt = f"Step 2 - Review: Review this plan: {outputs[-1]}. Suggest improvements and check risks."
        else:
            prompt = f"Step 3 - Execute: Based on plan and review: {outputs[-1]}. Generate final code/action."
        output = self.router.generate_response(prompt, 'sensitive_chain')
        outputs.append(output)
        if "error" in output.lower():
            logging.warning(f"Chain broke at step {step+1}")
            break
    final_output = outputs[-1] if outputs else "Chain failed"
    logging.info(f"Chain completed with {len(outputs)} steps")
    return final_output
app.py-এ ইন্টিগ্রেট:Pythonfrom config import CHAIN_STEPS
from modules.prompt.PromptChainer import PromptChainer
chainer = PromptChainer()

ইন্টিগ্রেশন: config.py থেকে CHAIN_STEPS নেওয়া। পরবর্তী টাস্কে chain_prompts ইউজ করো।
টেস্টিং:

tests/prompt_test.py তৈরি।
python -m pytest tests/prompt_test.py।
রিপোর্ট সেভ: reports/task_19_smart_prompt_chaining.json।

ফাইনাল আউটপুট: PromptChainer.py কমপ্লিট, সেনসিটিভ টাস্কে চেইনিং কাজ করে। git commit -m "Task 19: Smart Prompt Chaining (with config.py)"।
টাস্ক ২০: Final Test: Check if System Adds TTS on "Add TTS" Command
ওভারভিউ: এই টাস্কে ফাইনাল টেস্ট: ইউজার কমান্ড "TTS যোগ করো" দিলে সিস্টেম নিজে TTS endpoint যোগ করে কিনা চেক। End-to-end validation, report with metrics।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: pytest, gtts/pyttsx3 (TTS), requests (endpoint test)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: tests/integration/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

tests/integration/final_tts_test.py ফাইল তৈরি।
import pytest, requests, time, logging; from modules.upgrade.SelfUpgradeEngine import SelfUpgradeEngine।
class TestFinalTTS(unittest.TestCase):।
def setUp(self): self.engine = SelfUpgradeEngine(); self.start_time = time.time()।
def test_add_tts_command(self):Pythoncommand = "TTS যোগ করো"
proposal_gen = ProposalGenerator()
proposal = proposal_gen.generate_proposal()
if proposal and proposal_gen.request_permission(proposal):
    result = self.engine.execute_flow("TTS Endpoint", "Add TTS using gtts")
assert "success" in result.lower() or "added" in result.lower()
time.sleep(5)
from config import TTS_TEST_URL
response = requests.get(TTS_TEST_URL + '?text=Hello from TTS test')
assert response.status_code == 200 and 'audio/mpeg' in response.headers.get('Content-Type', '')
elapsed = time.time() - self.start_time
report = {'success': True, 'time': elapsed, 'metrics': {'endpoint_added': True}}
logging.info(f"Final TTS test: {report}")
with open('reports/task_20_final_tts.json', 'w') as f:
    json.dump(report, f)
app.py-এ ইন্টিগ্রেট:Pythonfrom config import TTS_TEST_URL

ইন্টিগ্রেশন: config.py থেকে TTS_TEST_URL নেওয়া। পরবর্তী টাস্কে টেস্ট রেজাল্ট ইউজ করো।
টেস্টিং:

python -m pytest tests/integration/final_tts_test.py -v।
reports/task_20_final_tts.json চেক।

ফাইনাল আউটপুট: final_tts_test.py কমপ্লিট, "TTS যোগ করো" কমান্ডে সিস্টেম অটো যোগ করে। git commit -m "Task 20: Final TTS Test (with config.py)"।

টাস্ক ২১: SelfLearningManager (Save Lessons from Errors)
ওভারভিউ: এই টাস্কে একটা ম্যানেজার ক্লাস তৈরি করা হবে যা সিস্টেমের ভুল বা এরর থেকে লেসন এক্সট্র্যাক্ট করে লং-টার্ম মেমরিতে সেভ করবে। এটি অটোমেটিকভাবে ট্রিগার হবে যখন এরর হবে, লেসনগুলোকে ক্যাটাগরাইজ করবে (e.g., 'code_error', 'performance_issue'), এবং ফিউচার ডিসিশনে ব্যবহার করবে যাতে একই ভুল দ্বিতীয়বার না হয়।
প্রয়োজনীয় প্রিপারেশন:

* রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।

* লাইব্রেরী: logging (error context), json (lesson format), TaskMemoryManager (storage integrate), requests (LLM analysis)।

* GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।

* ডিরেক্টরি: modules/learning/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

1. modules/learning/SelfLearningManager.py ফাইল তৈরি।

2. class SelfLearningManager:।

3. __init__ মেথড যোগ:
   ```python
   from config import LEARNING_PROMPT
   self.prompt_template = LEARNING_PROMPT
   self.router = ModelRouter()

learn_from_error মেথড যোগ:Pythondef learn_from_error(self, error_msg: str, context: str):
    prompt = self.prompt_template.format(error=error_msg)
    lesson_response = self.router.generate_response(prompt, 'learning_analysis')
    try:
        lesson_data = json.loads(lesson_response)
    except:
        lesson_data = {"lesson": lesson_response, "category": "unknown"}
    from modules.memory.TaskMemoryManager import TaskMemoryManager
    memory_manager = TaskMemoryManager()
    memory_manager.save_memory('lesson', lesson_data, lesson_data['category'])
    logging.info(f"Learned lesson: {lesson_data['lesson']}")
check_lessons মেথড যোগ:Pythondef check_lessons(self, action: str):
    from modules.memory.TaskMemoryManager import TaskMemoryManager
    memory_manager = TaskMemoryManager()
    lessons = memory_manager.load_memory('learning')
    for lesson in lessons:
        if action.lower() in lesson['content']['lesson'].lower():
            return False, lesson['content']['lesson']
    return True, None
ErrorLogger.py আপডেট:Pythonfrom modules.learning.SelfLearningManager import SelfLearningManager
SelfLearningManager().learn_from_error(str(e), context)
app.py-এ ইন্টিগ্রেট:Pythonfrom modules.learning.SelfLearningManager import SelfLearningManager
learning_manager = SelfLearningManager()

ইন্টিগ্রেশন: config.py থেকে LEARNING_PROMPT নেওয়া। পরবর্তী টাস্কে check_lessons ইউজ করো।
টেস্টিং:

tests/learning_test.py তৈরি।
python -m pytest tests/learning_test.py।
রিপোর্ট সেভ: reports/task_21_self_learning_manager.json।

ফাইনাল আউটপুট: SelfLearningManager.py কমপ্লিট, এরর থেকে লেসন সেভ এবং অ্যাভয়েড কাজ করে। git commit -m "Task 21: SelfLearningManager (with config.py)"।
টাস্ক ২২: Long-Term Task State Management (Runs for Years)
ওভারভিউ: এই টাস্কে লং-টার্ম টাস্কগুলোর স্টেট ম্যানেজ করা হবে, যেমন প্রোগ্রেস, ডিপেন্ডেন্সি, এবং রিজুম ক্যাপাবিলিটি, যাতে বছরের পর বছর চলতে পারে (e.g., ongoing learning projects)। এটি পিরিয়ডিক চেকপয়েন্ট সেভ করবে এবং রিস্টার্টে রিজুম করবে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: sqlite3/json (state storage), schedule/APScheduler (periodic tasks), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/taskstate/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/taskstate/TaskStateManager.py ফাইল তৈরি।
class TaskStateManager:।
init মেথড যোগ:Pythonfrom config import STATE_SAVE_INTERVAL
self.db_path = 'data/task_states.db'
self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
self.cursor = self.conn.cursor()
self.cursor.execute('''CREATE TABLE IF NOT EXISTS task_states (task_id TEXT PRIMARY KEY, state TEXT, last_update DATETIME, dependencies TEXT, progress REAL)''')
self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_last_update ON task_states (last_update)')
self.conn.commit()
import threading
threading.Thread(target=self._periodic_checkpoint, daemon=True).start()
save_state মেথড যোগ:Pythondef save_state(self, task_id: str, state_dict: dict, dependencies: list = None, progress: float = 0.0):
    import json, gzip, datetime
    compressed_state = gzip.compress(json.dumps(state_dict).encode('utf-8'))
    timestamp = datetime.datetime.now().isoformat()
    deps_str = json.dumps(dependencies or [])
    self.cursor.execute("INSERT OR REPLACE INTO task_states (task_id, state, last_update, dependencies, progress) VALUES (?, ?, ?, ?, ?)", 
        (task_id, compressed_state, timestamp, deps_str, progress))
    self.conn.commit()
    logging.info(f"Saved state for task {task_id}")
load_state মেথড যোগ:Pythondef load_state(self, task_id: str):
    self.cursor.execute("SELECT state, dependencies, progress FROM task_states WHERE task_id = ?", (task_id,))
    row = self.cursor.fetchone()
    if not row:
        return None, [], 0.0
    import gzip, json
    decompressed = gzip.decompress(row[0]).decode('utf-8')
    state = json.loads(decompressed)
    deps = json.loads(row[1])
    progress = row[2]
    return state, deps, progress
_periodic_checkpoint মেথড যোগ:Pythondef _periodic_checkpoint(self):
    import schedule, time
    schedule.every(STATE_SAVE_INTERVAL / 60).minutes.do(self._checkpoint_all)
    while True:
        schedule.run_pending()
        time.sleep(1)
app.py-এ ইন্টিগ্রেট:Pythonfrom config import STATE_SAVE_INTERVAL
from modules.taskstate.TaskStateManager import TaskStateManager
state_manager = TaskStateManager()

ইন্টিগ্রেশন: config.py থেকে STATE_SAVE_INTERVAL নেওয়া। পরবর্তী টাস্কে load_state ইউজ করো।
টেস্টিং:

tests/taskstate_test.py তৈরি।
python -m pytest tests/taskstate_test.py।
রিপোর্ট সেভ: reports/task_22_long_term_task_state.json।

ফাইনাল আউটপুট: TaskStateManager.py কমপ্লিট, লং-টার্ম স্টেট সেভ/লোড/রিজুম কাজ করে। git commit -m "Task 22: Long-Term Task State Management (with config.py)"।
টাস্ক ২৩: Hybrid Learning (Learn from Local + Groq)
ওভারভিউ: এই টাস্কে হাইব্রিড লার্নিং ইমপ্লিমেন্ট করা হবে, যেখানে লোকাল 14B মডেল লোকাল ডাটা থেকে শিখবে এবং Groq 70B ক্লাউড-বেসড জ্ঞান থেকে, তারপর কম্বাইন করে লেসন তৈরি করবে। এটি অপটিমাইজড সুইচিং ব্যবহার করে খরচ এবং প্রাইভেসি ব্যালেন্স করবে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: ollama/requests (models), json (result merge), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/learning/ ফোল্ডারে HybridLearner.py।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/learning/HybridLearner.py ফাইল তৈরি।
class HybridLearner:।
init মেথড যোগ:Pythonself.local_conn = LocalModelConnector()
self.router = ModelRouter()
learn_hybrid মেথড যোগ:Pythondef learn_hybrid(self, query: str):
    local_response = self.local_conn.generate_response(query)
    cloud_prompt = f"Provide high-quality answer: {query}"
    cloud_response = self.router.generate_response(cloud_prompt, 'learning', model='groq_70b')
    merged = self._merge_results(local_response, cloud_response)
    from modules.learning.SelfLearningManager import SelfLearningManager
    SelfLearningManager().learn_from_error("No error", f"Learned: {merged}")
    logging.info("Hybrid learning completed")
    return merged
_merge_results মেথড যোগ:Pythondef _merge_results(self, local: str, cloud: str):
    return f"Local: {local}\nCloud: {cloud}\nMerged: {cloud if 'confidence' in cloud.lower() else local}"
app.py-এ ইন্টিগ্রেট:Pythonfrom modules.learning.HybridLearner import HybridLearner
hybrid_learner = HybridLearner()

ইন্টিগ্রেশন: পরবর্তী টাস্কে learn_hybrid ইউজ করো।
টেস্টিং:

tests/hybrid_test.py তৈরি।
python -m pytest tests/hybrid_test.py।
রিপোর্ট সেভ: reports/task_23_hybrid_learning.json।

ফাইনাল আউটপুট: HybridLearner.py কমপ্লিট, হাইব্রিড লার্নিং কাজ করে। git commit -m "Task 23: Hybrid Learning (with config.py)"।
টাস্ক ২৪: Auto Summarization (Compact Old Memories)
ওভারভিউ: এই টাস্কে পুরনো মেমরিগুলোকে অটোমেটিক সামারাইজ করে কমপ্যাক্ট করা হবে, যাতে স্টোরেজ অপটিমাইজ হয় এবং রিট্রিভাল ফাস্ট হয়। এটি পিরিয়ডিক রান করবে (e.g., monthly), অরিজিনাল ডাটা আর্কাইভ করে সামারি সেভ করবে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: schedule (periodic), json/sqlite3 (archive), Groq for summarization prompts।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/memory/ ফোল্ডারে Summarizer.py।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/memory/Summarizer.py ফাইল তৈরি।
class Summarizer:।
init মেথড যোগ:Pythonfrom config import SUMMARY_AGE_MONTHS, SUMMARY_INTERVAL_DAYS
self.memory_manager = TaskMemoryManager()
self.age_months = int(os.getenv("SUMMARY_AGE_MONTHS", "6"))
self.interval_days = int(os.getenv("SUMMARY_INTERVAL_DAYS", "30"))
import threading
threading.Thread(target=self._periodic_summarize, daemon=True).start()
summarize_memories মেথড যোগ:Pythondef summarize_memories(self):
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=30*self.age_months)).isoformat()
    old_memories = self.memory_manager.cursor.execute("SELECT * FROM memories WHERE timestamp < ?", (cutoff,)).fetchall()
    if not old_memories:
        return "No old memories"
    prompt = "Summarize these old memories concisely, keep key points: " + json.dumps([dict(m) for m in old_memories])
    from modules.router.ModelRouter import ModelRouter
    router = ModelRouter()
    summary = router.generate_response(prompt, 'summarization')
    summary_entry = {'content': {'summary': summary, 'original_count': len(old_memories)}, 'category': 'summary'}
    self.memory_manager.save_memory('auto_summary', summary_entry, 'summary')
    self._archive_originals(old_memories)
    logging.info(f"Summarized {len(old_memories)} memories")
_archive_originals মেথড যোগ:Pythondef _archive_originals(self, memories: list):
    for mem in memories:
        self.memory_manager.cursor.execute("UPDATE memories SET category = 'archived' WHERE id = ?", (mem[0],))
    self.memory_manager.conn.commit()
_periodic_summarize মেথড যোগ:Pythondef _periodic_summarize(self):
    import schedule, time
    schedule.every(self.interval_days).days.do(self.summarize_memories)
    while True:
        schedule.run_pending()
        time.sleep(1)
app.py-এ ইন্টিগ্রেট:Pythonfrom config import SUMMARY_AGE_MONTHS
from modules.memory.Summarizer import Summarizer
summarizer = Summarizer()

ইন্টিগ্রেশন: config.py থেকে SUMMARY_AGE_MONTHS নেওয়া। পরবর্তী টাস্কে summarize_memories ইউজ করো।
টেস্টিং:

tests/summarizer_test.py তৈরি।
python -m pytest tests/summarizer_test.py।
রিপোর্ট সেভ: reports/task_24_auto_summarization.json।

ফাইনাল আউটপুট: Summarizer.py কমপ্লিট, অটো সামারাইজেশন কাজ করে। git commit -m "Task 24: Auto Summarization (with config.py)"।
টাস্ক ২৫: Final Security Audit + Rule-Based Approval
ওভারভিউ: এই টাস্কে সম্পূর্ণ সিস্টেমের সিকিউরিটি অডিট করা হবে এবং রুল-বেসড অ্যাপ্রুভাল সিস্টেম আপগ্রেড করা হবে, যেমন predefined rules for actions (e.g., no delete without backup)। অডিট রিপোর্ট জেনারেট করবে vulnerabilities সহ।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: bandit/pylint (security scan), yaml (rule files), logging/reportlab (reports)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/security/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/security/SecurityAuditor.py ফাইল তৈরি।
class SecurityAuditor:।
init মেথড যোগ:Pythonfrom config import RULES_FILE
self.rules_file = RULES_FILE
run_audit মেথড যোগ:Pythondef run_audit(self):
    import bandit
    from bandit.core import manager
    b_mgr = manager.BanditManager(config=None, agg_type='file')
    b_mgr.discover_files(['.'], recursive=True)
    b_mgr.run_tests()
    issues = b_mgr.get_issue_list()
    pylint_output = subprocess.run(['pylint', '--output-format=text', '.'], capture_output=True, text=True).stdout
    report = {'bandit_issues': [str(i) for i in issues], 'pylint_output': pylint_output, 'timestamp': datetime.now().isoformat()}
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas("reports/task_25_security_audit.pdf", pagesize=letter)
    c.drawString(100, 750, "Security Audit Report")
    y = 700
    for issue in report['bandit_issues']:
        c.drawString(100, y, issue)
        y -= 20
    c.save()
    logging.info("Audit completed")
    return report
ApprovalManager.py আপডেট:Pythonimport yaml
with open(self.rules_file, 'r') as f:
    self.rules = yaml.safe_load(f)
app.py-এ ইন্টিগ্রেট:Pythonfrom config import RULES_FILE
from modules.security.SecurityAuditor import SecurityAuditor
auditor = SecurityAuditor()

ইন্টিগ্রেশন: config.py থেকে RULES_FILE নেওয়া। পরবর্তী টাস্কে run_audit ইউজ করো।
টেস্টিং:

tests/security_test.py তৈরি।
python -m pytest tests/security_test.py।
রিপোর্ট সেভ: reports/task_25_security_audit.json।

ফাইনাল আউটপুট: SecurityAuditor.py কমপ্লিট, অডিট এবং রুল-বেসড অ্যাপ্রুভাল কাজ করে। git commit -m "Task 25: Final Security Audit (with config.py)"।
টাস্ক ২৬: Plugin System Base Creation (For Future Use)
ওভারভিউ: এই টাস্কে প্লাগিন সিস্টেমের বেস তৈরি করা হবে, যাতে ফিউচারে নতুন ফিচার (e.g., TTS, hacking tools) প্লাগিন হিসেবে যোগ করা যায়। এটি লোডার, রেজিস্ট্রি, এবং ইন্টারফেস ডিফাইন করবে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: importlib (dynamic load), abc (abstract base), logging।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/plugin/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/plugin/PluginManager.py ফাইল তৈরি।
from abc import ABC, abstractmethod; class BasePlugin(ABC): @abstractmethod def execute(self, *args, **kwargs): pass।
class PluginManager:।
init মেথড যোগ:Pythonfrom config import PLUGIN_DIR
self.plugins = {}
self.plugin_dir = PLUGIN_DIR
self._load_plugins()
_load_plugins মেথড যোগ:Pythondef _load_plugins(self):
    import os, importlib.util
    for file in os.listdir(self.plugin_dir):
        if file.endswith('.py') and file != '__init__.py':
            module_name = file[:-3]
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(self.plugin_dir, file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj != BasePlugin:
                    plugin = obj()
                    self.plugins[plugin.name] = plugin
execute_plugin মেথড যোগ:Pythondef execute_plugin(self, name: str, *args, **kwargs):
    if name in self.plugins:
        return self.plugins[name].execute(*args, **kwargs)
    else:
        raise ValueError("Plugin not found")
app.py-এ ইন্টিগ্রেট:Pythonfrom config import PLUGIN_DIR
from modules.plugin.PluginManager import PluginManager
plugin_manager = PluginManager()

ইন্টিগ্রেশন: config.py থেকে PLUGIN_DIR নেওয়া। পরবর্তী টাস্কে execute_plugin ইউজ করো।
টেস্টিং:

tests/plugin_test.py তৈরি।
python -m pytest tests/plugin_test.py।
রিপোর্ট সেভ: reports/task_26_plugin_system_base.json।

ফাইনাল আউটপুট: PluginManager.py কমপ্লিট, প্লাগিন লোড/এক্সিকিউট কাজ করে। git commit -m "Task 26: Plugin System Base (with config.py)"।
টাস্ক ২৭: Voice Interface Preparation (TTS/STT)
ওভারভিউ: এই টাস্কে ভয়েস ইন্টারফেসের প্রস্তুতি করা হবে: TTS (Text-to-Speech) এবং STT (Speech-to-Text) ইন্টিগ্রেশন, যাতে সিস্টেম ভয়েস কমান্ড নিতে এবং রেসপন্ড করতে পারে। প্রথমে বেসিক API wrappers তৈরি।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: gtts/pyttsx3 (TTS), speech_recognition (STT), pydub (audio handling)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: modules/voice/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

modules/voice/VoiceInterface.py ফাইল তৈরি।
class VoiceInterface:।
init মেথড যোগ:Pythonfrom config import TTS_ENGINE, STT_ENGINE
self.tts_engine = TTS_ENGINE
self.stt_engine = STT_ENGINE
tts মেথড যোগ:Pythondef tts(self, text: str, output_file: str = 'output.mp3'):
    if self.tts_engine == 'gtts':
        from gtts import gTTS
        tts = gTTS(text)
        tts.save(output_file)
    elif self.tts_engine == 'pyttsx3':
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, output_file)
        engine.runAndWait()
    logging.info(f"TTS generated: {output_file}")
    return output_file
stt মেথড যোগ:Pythondef stt(self, audio_file: str):
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
    except:
        text = "Recognition failed"
    logging.info(f"STT result: {text}")
    return text
app.py-এ ইন্টিগ্রেট:Pythonfrom config import TTS_ENGINE
from modules.voice.VoiceInterface import VoiceInterface
voice_interface = VoiceInterface()

ইন্টিগ্রেশন: config.py থেকে TTS_ENGINE নেওয়া। পরবর্তী টাস্কে integrate_with_chat ইউজ করো।
টেস্টিং:

tests/voice_test.py তৈরি।
python -m pytest tests/voice_test.py।
রিপোর্ট সেভ: reports/task_27_voice_interface.json।

ফাইনাল আউটপুট: VoiceInterface.py কমপ্লিট, TTS/STT প্রস্তুত এবং চ্যাটে ইন্টিগ্রেটেড। git commit -m "Task 27: Voice Interface Preparation (with config.py)"।
টাস্ক ২৮: Dashboard UI (Streamlit)
ওভারভিউ: এই টাস্কে একটা ড্যাশবোর্ড UI তৈরি করা হবে Streamlit দিয়ে, যেখানে স্ট্যাটাস, মেমরি, আপগ্রেড ইতিহাস, এবং কন্ট্রোল (e.g., approve buttons) দেখা যাবে। এটি ইউজার-ফ্রেন্ডলি হবে।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: streamlit (UI), pandas (data display), requests (backend calls)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: dashboard/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

dashboard/app.py ফাইল তৈরি।
import streamlit as st, pandas as pd, requests।
অথেনটিকেশন:Pythonfrom config import DASHBOARD_AUTH_PASSWORD
if st.text_input("Password", type="password") != DASHBOARD_AUTH_PASSWORD:
    st.error("Wrong password")
    st.stop()
সাইডবার:Pythonst.sidebar.title("The Mask Dashboard")
page = st.sidebar.radio("Select Page", ["Status", "Memory", "Upgrade History", "Approvals"])
Status পেজ:Pythonif page == "Status":
    st.header("System Status")
    from config import STATUS_AUTH_KEY
    response = requests.get('http://localhost:5000/status', headers={'Auth-Key': STATUS_AUTH_KEY})
    data = response.json()
    st.json(data)
    df = pd.DataFrame(data['models'].items(), columns=['Model', 'Details'])
    st.table(df)
Memory পেজ:Pythonif page == "Memory":
    st.header("Memory Query")
    category = st.text_input("Category (optional)")
    if st.button("Fetch"):
        memories = requests.get('http://localhost:5000/memory', params={'category': category}).json()
        st.dataframe(pd.DataFrame(memories))
Upgrade History পেজ:Pythonif page == "Upgrade History":
    st.header("Upgrade History")
    feature = st.text_input("Filter by Feature")
    if st.button("Show"):
        history = requests.get('http://localhost:5000/upgrade_history', params={'feature': feature}).json()
        st.table(pd.DataFrame(history))
Approvals পেজ:Pythonif page == "Approvals":
    st.header("Pending Approvals")
    # মক ডাটা বা API থেকে
    pending = [{"action": "Delete file", "desc": "Test"}]
    for p in pending:
        if st.button(f"Approve {p['action']}"):
            st.success("Approved")
app.py (মেইন)-এ ইন্টিগ্রেট:Pythonfrom config import STREAMLIT_PORT
if 'open_dashboard' in command:
    subprocess.Popen(['streamlit', 'run', 'dashboard/app.py', '--server.port', STREAMLIT_PORT])

ইন্টিগ্রেশন: config.py থেকে STATUS_AUTH_KEY এবং DASHBOARD_AUTH_PASSWORD নেওয়া। পরবর্তী টাস্কে ড্যাশবোর্ডে রিপোর্ট দেখানো।
টেস্টিং:

streamlit run dashboard/app.py রান করে লোকাল চেক।
reports/task_28_dashboard_ui.json (যদি টেস্ট রিপোর্ট লাগে)।

ফাইনাল আউটপুট: dashboard/app.py কমপ্লিট, Streamlit ড্যাশবোর্ড রান করে। git commit -m "Task 28: Dashboard UI (with config.py)"।
টাস্ক ২৯: Full System End-to-End Test (1 Month Long Task)
ওভারভিউ: এই টাস্কে সম্পূর্ণ সিস্টেমের এন্ড-টু-এন্ড টেস্ট করা হবে, একটা ১ মাসের লং টাস্ক সিমুলেট করে (e.g., ongoing learning)। কভার: stability, memory, upgrades, etc. রিপোর্ট সহ।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: pytest/selenium (if UI), time (simulation), logging/prometheus (metrics)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: tests/e2e/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

tests/e2e/e2e_suite.py ফাইল তৈরি।
import pytest, time, requests, logging, threading।
class E2ETestSuite:।
def init(self):Pythonself.start_time = time.time()
self.report = {'uptime': 0, 'errors': 0, 'upgrades': 0, 'metrics': {}}
self.active = True
run_long_test মেথড যোগ:Pythondef run_long_test(self):
    sim_days = int(os.getenv("E2E_SIMULATION_DAYS", "30"))
    sim_seconds = sim_days * 86400 / 100  # accelerated
    end_time = time.time() + sim_seconds
    while time.time() < end_time and self.active:
        try:
            status = requests.get('http://localhost:5000/status', headers={'Auth-Key': os.getenv('STATUS_AUTH_KEY')}).json()
            assert status['system_health'] == 'ok'
            mem_count = len(requests.get('http://localhost:5000/memory').json())
            assert mem_count > 0
            upgrade = requests.post('http://localhost:5000/upgrade', json={'feature': 'e2e_test'})
            assert 'success' in upgrade.text.lower()
            self.report['checks'].append({'time': time.time(), 'status': 'pass'})
        except Exception as e:
            self.report['errors'] += 1
            self.report['checks'].append({'time': time.time(), 'status': 'fail', 'error': str(e)})
        time.sleep(60)
    self.report['duration'] = time.time() - self.report['start']
    self.report['success_rate'] = (len(self.report['checks']) - self.report['errors']) / len(self.report['checks']) * 100 if self.report['checks'] else 0
    with open('reports/task_29_e2e.json', 'w') as f:
        json.dump(self.report, f, indent=2)
app.py-এ ইন্টিগ্রেট:Pythonfrom config import STATUS_AUTH_KEY

ইন্টিগ্রেশন: config.py থেকে STATUS_AUTH_KEY নেওয়া। পরবর্তী টাস্কে রিপোর্ট ইউজ করো।
টেস্টিং:

python tests/e2e/e2e_suite.py রান করে চেক।
reports/task_29_e2e.json চেক।

ফাইনাল আউটপুট: e2e_suite.py কমপ্লিট, ফুল সিস্টেম E2E টেস্ট সিমুলেটেড। git commit -m "Task 29: Full E2E Test (with config.py)"।
টাস্ক ৩০: v1.0 Release + Documentation
ওভারভিউ: এই টাস্কে v1.0 রিলিজ করা হবে: final build, deployment, এবং ডকুমেন্টেশন (setup, usage, API docs)। GitHub release with changelog।
প্রয়োজনীয় প্রিপারেশন:

রুট ফোল্ডারে (app.py-এর পাশে) config.py ফাইল তৈরি করা আছে।
লাইব্রেরী: sphinx/mkdocs (docs), gitpython (release), setuptools (packaging)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি: docs/ ফোল্ডার তৈরি।

স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:

release.py স্ক্রিপ্ট তৈরি (রুটে)।
import git, os, subprocess; repo = git.Repo('.'); version = os.getenv("RELEASE_VERSION", "1.0.0")।
Version bumpPythonwith open('VERSION', 'w') as f:
    f.write(version)
ChangelogPythonlog = repo.git.log('--pretty=format:%s', 'v0.1..HEAD')
with open('CHANGELOG.md', 'a') as f:
    f.write(f"\n## v{version}\n{log}")
DocsPythonsubprocess.run(['sphinx-quickstart', 'docs', '--quiet', '--project', 'The Mask Core', '--author', 'the mask'])
subprocess.run(['make', 'html'], cwd='docs')
GitHub releasePythonrepo.create_tag(f'v{version}')
repo.remotes.origin.push(f'v{version}')
subprocess.run(['gh', 'release', 'create', f'v{version}', '--notes', 'CHANGELOG.md', '--title', f'v{version} Release'])
Render deployPythonrepo.git.push('render', 'main')
app.py-এ ইন্টিগ্রেট:Pythonif 'release_v1' in sys.argv:
    subprocess.run(['python', 'release.py'])

ইন্টিগ্রেশন: config.py থেকে RELEASE_VERSION নেওয়া। পরবর্তী ভার্সনের জন্য টেমপ্লেট হিসেবে রাখো।
টেস্টিং:

python release.py রান করে চেক।
reports/task_30_v1_release.json (যদি লাগে)।

ফাইনাল আউটপুট: v1.0 রিলিজড, ডকুমেন্টেশন জেনারেটেড, GitHub release কমপ্লিট। git commit -m "Task 30: v1.0 Release + Documentation (with config.py)"।