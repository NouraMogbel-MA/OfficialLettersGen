# -*- coding: utf-8 -*-
# Demo: write a Saudi administrative letter with the methods of the thesis.
# run:  streamlit run app/app.py
# Methods 3-5 use ALLaM-7B-Instruct-preview (downloaded from Hugging Face on first use; a GPU with about 16 GB is needed).
# Method 2 (m2m100 trained on the letter information) is offered when its weights are in the folder named by M2M_MODEL_DIR.
import os, json, sys
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompting as P

ROOT = os.path.dirname(HERE)
ALLAM = os.environ.get('ALLAM_MODEL', 'humain-ai/ALLaM-7B-Instruct-preview')
M2M_DIR = os.environ.get('M2M_MODEL_DIR', os.path.join(ROOT, 'models', 'm2m100_fields'))
METHODS = {'الأمر دون أمثلة': 'zero', 'الأمر بمثالين': 'few', 'الأمر اللغوي': 'ling'}
if os.path.exists(os.path.join(M2M_DIR, 'config.json')):
    METHODS['التدريب على المعلومات (فيسبوك m2m100)'] = 'm2m'

st.set_page_config(page_title='OfficialLettersGen', layout='wide')
st.markdown('<style>body, .stTextArea textarea, .stTextInput input, .stMarkdown {direction: rtl; text-align: right;}</style>',
            unsafe_allow_html=True)
st.title('كتابة الخطاب الإداري')


@st.cache_data
def data():
    return json.load(open(os.path.join(ROOT, 'data', 'inputs.json'), encoding='utf-8'))


@st.cache_resource
def allam():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(ALLAM)
    model = AutoModelForCausalLM.from_pretrained(ALLAM, torch_dtype=torch.bfloat16, device_map='auto').eval()
    return tok, model


@st.cache_resource
def m2m():
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    tok = M2M100Tokenizer.from_pretrained(M2M_DIR, src_lang='ar', tgt_lang='ar')
    model = M2M100ForConditionalGeneration.from_pretrained(M2M_DIR).eval()
    return tok, model


def generate(method, text):
    import torch
    if method == 'm2m':
        tok, model = m2m()
        x = tok(' '.join(text.split()), return_tensors='pt', truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            o = model.generate(**x, max_length=1024, min_length=128, do_sample=True, temperature=0.3, top_k=50, top_p=0.95,
                               forced_bos_token_id=tok.get_lang_id('ar'))
        return tok.decode(o[0], skip_special_tokens=True)
    tok, model = allam()
    s = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
    x = tok(s, return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o = model.generate(**x, do_sample=True, temperature=0.3, top_p=0.95, top_k=50, max_new_tokens=1024)
    return tok.decode(o[0][x['input_ids'].shape[1]:], skip_special_tokens=True).strip()


D = data()
with st.sidebar:
    st.header('مثال من خطابات الاختبار')
    k = st.number_input('رقم الخطاب', 0, len(D['test']) - 1, 0)
    if st.button('املأ المعلومات من هذا الخطاب'):
        st.session_state['example'] = D['test'][int(k)]
    method_name = st.radio('الطريقة', list(METHODS))

ex = st.session_state.get('example')
if ex:
    st.info('المعلومات مأخوذة من خطاب الاختبار، ويمكن تعديلها.')
    inp = st.text_area('المعلومات التي تُعطى للنموذج', ex['input'], height=320)
    ty = ex['qtype']
else:
    c1, c2 = st.columns(2)
    with c1:
        ty = st.selectbox('نوع الخطاب', P.TYPES)
        subject = st.text_input('الموضوع')
        position = st.text_input('منصب المخاطَب', placeholder='وزير التعليم')
        entity = st.text_input('جهة المخاطَب', placeholder='وزارة التعليم')
        ref_no = st.text_input('رقم الخطاب السابق، إن وُجد')
        ref_date = st.text_input('تاريخه')
    with c2:
        points = st.text_area('النقاط الأساسية، نقطة في كل سطر', height=150)
        request = st.text_input('المطلوب')
        contact = st.text_input('بيانات التواصل')
        sender = st.text_input('المرسل')
    info = P.info_text({'type': ty, 'subject': subject, 'position': position, 'entity': entity, 'ref_no': ref_no,
                        'ref_date': ref_date, 'points': points.split('\n'), 'request': request, 'contact': contact, 'sender': sender})
    inp = P.model_input(P.question(ty, subject), info)

method = METHODS[method_name]
text = inp if method == 'm2m' else P.prompt(method, ty, inp, D['train'])
with st.expander('الأمر الذي يُرسل إلى النموذج'):
    st.text(text)
if st.button('اكتب الخطاب', type='primary'):
    with st.spinner('يكتب النموذج الخطاب...'):
        st.text_area('الخطاب المولَّد', generate(method, text), height=420)
    st.caption('الخطاب المولَّد مسودة، ويجب مراجعة أسمائه وأرقامه قبل استعماله.')
