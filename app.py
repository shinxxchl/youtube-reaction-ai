import os
import re
import html
from collections import Counter
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='YouTube Reaction AI', page_icon='🎬', layout='wide')
st.markdown('''<style>
.block-container {max-width:1050px;padding-top:2rem}
h1 {letter-spacing:-1px}
[data-testid="stMetric"] {background:#f5f6fa;border-radius:14px;padding:16px}
</style>''', unsafe_allow_html=True)

STOP = set('영상 댓글 사람 생각 내용 정말 진짜 너무 그냥 약간 이번 해당 이것 저것 여기 저기 자신 정도 경우 때문 관련 부분 모두 지금 오늘 어제 내일 제가 나는 내가 너가 우리 여러분 그리고 하지만 그래서 그러니까 또한 이런 그런 어떤 있는 없는 하는 되는 했던 위해 대한 너무 그냥 아주 매우 계속 제발 역시 역시 감사합니다 감사 해요 입니다 있어요 있네요 있나요 같아요 하네요 않아요 이거 저거 그거 거기 저기 때문 대해 통해 영상분 댓글들 영상이 채널 유튜브 youtube shorts video https http www com'.split())
API_URL = 'https://www.googleapis.com/youtube/v3'

def extract_video_id(value):
    value = value.strip()
    if re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
        return value
    parsed = urlparse(value if '://' in value else 'https://' + value)
    domain = (parsed.hostname or '').lower()
    if domain in ('youtu.be', 'www.youtu.be'):
        candidate = parsed.path.strip('/').split('/')[0]
    elif domain in ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com'):
        segments = parsed.path.strip('/').split('/')
        if parsed.path.rstrip('/') == '/watch':
            candidate = parse_qs(parsed.query).get('v', [''])[0]
        elif len(segments) >= 2 and segments[0] in ('shorts', 'live', 'embed'):
            candidate = segments[1]
        else:
            candidate = ''
    else:
        candidate = ''
    return candidate if re.fullmatch(r'[A-Za-z0-9_-]{11}', candidate) else None


def api_get(endpoint, key, params):
    try:
        r = requests.get(API_URL + '/' + endpoint, params={**params, 'key':key}, timeout=20)
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError('유튜브 API 연결에 실패했어요. 인터넷 연결을 확인해 주세요.') from exc
    if not r.ok:
        reason = (data.get('error', {}).get('errors') or [{}])[0].get('reason', '')
        if reason == 'commentsDisabled':
            raise RuntimeError('댓글이 비활성화된 영상이에요.')
        if reason in ('quotaExceeded', 'dailyLimitExceeded'):
            raise RuntimeError('YouTube API 일일 사용 한도가 초과됐어요.')
        if r.status_code == 403:
            raise RuntimeError('API 접근이 거부됐어요. YouTube Data API v3 활성화와 API 키 제한을 확인하세요.')
        raise RuntimeError(data.get('error', {}).get('message', 'YouTube API 오류가 발생했습니다.'))
    return data

@st.cache_data(ttl=1200, show_spinner=False)
def load_comments(video_id, key, limit):
    videos = api_get('videos', key, {'part':'snippet', 'id':video_id, 'fields':'items(snippet(title,channelTitle))'})
    items = videos.get('items', [])
    if not items:
        raise RuntimeError('영상을 찾을 수 없어요. 공개 영상인지 확인해 주세요.')
    title = items[0]['snippet']['title']
    channel = items[0]['snippet']['channelTitle']
    result, token = [], None
    while len(result) < limit:
        params = {'part':'snippet','videoId':video_id,'maxResults':min(100,limit-len(result)),
                  'order':'time','textFormat':'plainText'}
        if token:
            params['pageToken'] = token
        data = api_get('commentThreads', key, params)
        for item in data.get('items', []):
            s = item['snippet']['topLevelComment']['snippet']
            result.append({'댓글':html.unescape(s.get('textDisplay','')),
                           '좋아요':s.get('likeCount',0), '작성 시각':s.get('publishedAt','')})
        token = data.get('nextPageToken')
        if not token or not data.get('items'):
            break
    return title, channel, result

@st.cache_resource(show_spinner='한국어 분석 모델을 준비하는 중...')
def get_classifier():
    from transformers import pipeline
    return pipeline('text-classification', model='daekeun-ml/koelectra-small-v3-nsmc', device=-1)

@st.cache_resource(show_spinner='한국어 키워드 분석기를 준비하는 중...')
def get_kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()

def classify(texts, threshold):
    classifier = get_classifier()
    result = []
    for start in range(0,len(texts),16):
        batch = [re.sub(r'\s+', ' ', t).strip()[:500] or ' ' for t in texts[start:start+16]]
        outputs = classifier(batch, truncation=True, max_length=256, batch_size=16)
        for output in outputs:
            raw = output['label'].upper()
            score = float(output['score'])
            if score < threshold:
                label = '판단 보류'
            elif raw in ('1','LABEL_1','POSITIVE','POS','긍정'):
                label = '긍정'
            elif raw in ('0','LABEL_0','NEGATIVE','NEG','부정'):
                label = '부정'
            else:
                label = '판단 보류'
            result.append((label, round(score,3)))
    return result

def keywords(texts, top=15):
    kiwi = get_kiwi()
    freq = Counter()
    for line in texts:
        for token in kiwi.tokenize(line[:2000]):
            word = token.form.lower()
            if token.tag.startswith('NN') and len(word)>=2 and word not in STOP and not word.isdigit():
                freq[word] += 1
    return freq.most_common(top)

st.title('🎬 YouTube Reaction AI')
st.write('유튜브 댓글을 AI로 분석해 **긍정·부정 반응과 핵심 키워드**를 알려드립니다.')
st.caption('한국어 댓글 중심 · 공개 영상의 최상위 댓글 · 최근 댓글부터 수집')

try:
    secret_key = st.secrets.get('YOUTUBE_API_KEY', '')
except (FileNotFoundError, KeyError):
    secret_key = ''
key = os.environ.get('YOUTUBE_API_KEY', '') or secret_key
if not key:
    st.warning('YouTube API 키 설정이 필요해요. Streamlit Secrets에 YOUTUBE_API_KEY를 등록하세요.')

with st.form('analyze'):
    url = st.text_input('분석할 영상 주소', placeholder='https://www.youtube.com/watch?v=...')
    left, right = st.columns(2)
    with left:
        limit = st.select_slider('수집할 댓글 수 (최대)', options=[50,100,200,300,500], value=200)
    with right:
        threshold = st.slider('판단 보류 기준 (AI 확신도)', 0.50, 0.95, 0.65, 0.05)
    submitted = st.form_submit_button('🔍 댓글 분석하기', type='primary', use_container_width=True)

if submitted:
    video_id = extract_video_id(url)
    if not video_id:
        st.error('올바른 유튜브 영상 주소를 입력해 주세요. (일반 영상, Shorts, youtu.be 지원)')
    elif not key:
        st.error('API 키가 설정되지 않았어요.')
    else:
        try:
            with st.spinner('유튜브 댓글을 수집하는 중...'):
                title, channel, rows = load_comments(video_id, key, limit)
            if not rows:
                st.warning('수집 가능한 공개 댓글이 없어요.')
            else:
                with st.spinner('AI가 댓글 감정을 분석하고 키워드를 추출하는 중...'):
                    predictions = classify([r['댓글'] for r in rows],threshold)
                    for row,(label,confidence) in zip(rows,predictions):
                        row['분석 결과'] = label
                        row['AI 확신도'] = confidence
                    all_words = keywords([r['댓글'] for r in rows])
                    negative_words = keywords([r['댓글'] for r in rows if r['분석 결과']=='부정'])
                st.session_state['result'] = {'video_id':video_id,'title':title,'channel':channel,
                    'rows':rows, 'all_words':all_words,'negative_words':negative_words,'limit':limit,'threshold':threshold}
        except Exception as exc:
            st.error(f'분석할 수 없어요: {exc}')

if 'result' in st.session_state:
    data = st.session_state['result']
    st.divider()
    st.subheader(data['title'])
    st.caption(f"채널: {data['channel']} · 분석 대상: 최근 공개 최상위 댓글 {len(data['rows']):,}개")
    frame = pd.DataFrame(data['rows'])
    counts = frame['분석 결과'].value_counts()
    total = len(frame)
    pos = int(counts.get('긍정',0)); neg = int(counts.get('부정',0)); unsure = int(counts.get('판단 보류',0))
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('분석 댓글',f'{total:,}개')
    c2.metric('😊 긍정',f'{pos/total:.1%}')
    c3.metric('😟 부정',f'{neg/total:.1%}')
    c4.metric('🤔 판단 보류',f'{unsure/total:.1%}')
    st.progress(pos/total, text=f'긍정 댓글 비율: {pos/total:.1%}')
    st.caption('판단 보류는 실제 중립 감정과 같지 않습니다. 이 AI는 긍정/부정 2분류 모델이며 확신도가 낮으면 판단을 보류합니다.')
    st.subheader('📊 감정 분석 결과')
    st.bar_chart(pd.DataFrame({'댓글 수':[pos,neg,unsure]},index=['긍정','부정','판단 보류']))
    a,b = st.columns(2)
    with a:
        st.subheader('🔑 전체 핵심 키워드')
        if data['all_words']:
            st.bar_chart(pd.DataFrame(data['all_words'],columns=['키워드','언급 횟수']).set_index('키워드'))
        else:
            st.info('추출된 키워드가 없습니다.')
    with b:
        st.subheader('⚠️ 부정 댓글 핵심 키워드')
        if data['negative_words']:
            st.bar_chart(pd.DataFrame(data['negative_words'],columns=['키워드','언급 횟수']).set_index('키워드'))
        else:
            st.info('부정 댓글에서 추출한 키워드가 없습니다.')
    st.subheader('💬 분석된 댓글 보기')
    selected = st.selectbox('댓글 필터', ['전체','긍정','부정','판단 보류'])
    shown = frame if selected=='전체' else frame[frame['분석 결과']==selected]
    st.dataframe(shown, hide_index=True, use_container_width=True)
    st.download_button('📥 결과 CSV 다운로드',frame.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"youtube_reactions_{data['video_id']}.csv",mime='text/csv')
    st.info('분석 범위는 영상의 모든 댓글이 아니라 수집한 최상위 댓글이며, 답글은 포함하지 않습니다. 감정 분류는 오분류될 수 있고 한국어 영화 리뷰 학습 모델이라 유튜브 댓글과 차이가 있을 수 있습니다.')
