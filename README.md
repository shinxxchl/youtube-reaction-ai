# YouTube Reaction AI

유튜브 영상 URL을 입력하면 최근 공개 최상위 댓글을 수집하고, 한국어 감정 분석 모델을 통해 긍정·부정 비율과 핵심 키워드를 보여주는 Streamlit 앱입니다.

## 포함 파일
- `app.py`: 웹앱 전체 코드
- `requirements.txt`: 설치 패키지
- `.gitignore`: 비밀 키 업로드 방지

## 배포하기
1. GitHub에 새 저장소를 만듭니다.
2. 이 폴더의 `app.py`, `requirements.txt`, `.gitignore`, `README.md`를 업로드합니다.
3. https://share.streamlit.io 에서 Create app → GitHub 저장소 선택 → Main file `app.py`.
4. Advanced settings → Secrets에 아래처럼 등록합니다(실제 키로 교체):

```toml
YOUTUBE_API_KEY = "YOUR_NEW_API_KEY"
```

5. Deploy를 누른 뒤 생성된 URL을 수행평가에 제출합니다.

## API 설정
Google Cloud Console에서 프로젝트 생성 → YouTube Data API v3 활성화 → 사용자 인증 정보 → API 키 생성. 키를 GitHub 코드나 README에 적지 마세요. 이전에 공개한 키는 폐기하고 재발급하는 편이 좋습니다.

## 한계
- 한국어 영화 리뷰로 학습한 **이진(긍정/부정)** 모델 `daekeun-ml/koelectra-small-v3-nsmc`를 활용합니다.
- `판단 보류`는 3분류 모델의 중립 라벨이 아니라, 이진 모델의 확신도가 낮을 때 붙인 별도 표기입니다.
- 최근 댓글 중 최대 500개, 답글 제외. 댓글을 전부 분석한 결과가 아닙니다.
- 모델을 첫 실행 시 다운로드하므로 배포 환경의 메모리와 설치 시간 제한에 영향을 받을 수 있습니다.
- 감정 분석은 풍자, 농담, 은어 등을 잘못 해석할 수 있습니다.
