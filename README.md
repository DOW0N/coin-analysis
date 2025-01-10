비트코인시세분석데이터및예측분석프로젝트- 김도원
preencoded.png
프로젝트개요
이프로젝트는비트코인시세데이터를MySQL 데이터베이스에서불러와분석하고, 향후시세를예측하는시스템을구축한것입니다. 
데이터는비트코인의과거가격및거래량정보를기반으로분석하고, Linear Regression 모델을사용하여비트코인시세의예측값을도
출합니다. 분석된데이터를통해사용자에게실시간시세및향후시세예측정보를제공합니다.
 preencoded.png
 2
사용된기술스택:
 • Python: 데이터분석및머신러닝모델링
• Pandas: 데이터처리및전처리
• Matplotlib: 데이터 시각화
• Scikit-learn: Linear Regression을 사용한 시세 예측 모델
• SQLAlchemy: MySQL 데이터베이스와의연결및데이터
조회
• MySQL: 데이터베이스관리시스템(가격및거래량데이터
저장)
 • github : 
https://github.com/DOW0N/coin-analysis.git
 preencoded.png
 3
주요기능
1. 데이터로딩및전처리:
 • MySQL에서비트코인및암호화폐관련데이터를쿼리하여Pandas DataFrame으로로딩
• 가격과거래량데이터의이상값및결측값처리
2. 시각화:
 • 비트코인가격의시간에따른변동을시각화하여사용자에게가격추이를직관적으로전달
• 가격예측을위해Linear Regression 모델을학습하고, 예측된가격을시각화하여향후시세를예측
3. 가격예측:
 • Linear Regression 모델을 사용하여향후일자에대한비트코인가격을예측
• 예측된데이터를시각적으로제공하여사용자가향후가격변동을예측할수있도록지원
preencoded.png
 4
데이터수집및전처리
1
 1. 데이터수집
데이터수집은주로외부데이터소스에서데이터를가져오는과정입니다. 이프로젝트에서는MySQL데이터베이스에서비트코인시세와관련된데이터를수집
2
데이터정제
결측값처리
데이터에서결측값(NaN)은모델학습에악영향을미칠수있으므로적절히처리해야합니다. dropna()를사용하여결측값이포함된행을삭제합니다.
이상값처리
가격데이터나거래량데이터는이상값을포함할수있습니다. 가격이너무높거나낮은값은이상값일수있으며, 이를처리하는방법은다양합니다. 예를들어, 가격이0보다작은값이나비정상적으로큰값은NaN으로처
리하거나삭제할수있습니다.
 3
데이터준비
이제데이터를예측모델에사용할수있는형식으로준비합니다. 예를들어, 가격예측을위한데이터는timestamp와price컬럼을사용할수있습니다.
 4
시각화(데이터확인)
전처리된데이터를시각화하여데이터의패턴이나이상값을확인할수있습니다. Matplotlib을사용하여가격변동을시각적으로표현합니다.
 preencoded.png
 5
SQL 쿼리
•WHERE timestamp >= '2025-01-01' – 2025년 이후의 데이터만 필터링합니다.
 •volume >= 100 – 거래량이 100 이상인 데이터만 조회합니다.
 •timestamp >= NOW() - INTERVAL 30 DAY – 최근 30일 동안의 데이터만 가져옵니다.
 •LIMIT 100 – 가장 최근 100개의 데이터를 조회해 예측 모델을 학습하는 데 사용됩니다.
 •AVG(price) AS avg_price – 날짜별로 가격의 평균을 계산합니다.
 •SUM(volume) AS total_volume – 해당 날짜의 총 거래량을 집계합니다.
 •GROUP BY DATE(timestamp) – 날짜별로 데이터를 그룹화하여 일별 평균을 산출합니다.
 •ORDER BY date – 날짜순으로 데이터를 정렬합니다. 
preencoded.png
 6
작업코드(일부)
 preencoded.png
 7
시각화그래프
비트코인가격변동그래프
• 목적: 이그래프는비트코인의역사적인가격변동을시간에따라시각적으로보
여줍니다.
 • X축: timestamp(날짜) -비트코인시세가기록된날짜.
 • Y축: price(가격) - 각날짜에해당하는비트코인의가격.
 • 그래프특징: 비트코인의가격변동을시간에따라표시하여시세가어떻게변화
했는지, 급격한상승이나하락이있었는지, 또는특정시점에패턴이존재하는지
확인할수있습니다.
날짜별비트코인가격예측값그래프
• 목적: 과거데이터를기반으로머신러닝모델을사용하여비트코인의미래가격
을예측하고, 이를시각적으로표시합니다.
 • X축: 예측된날짜(future_dates) -미래날짜를나타냅니다. 이날짜들은비트코
인가격예측을위해생성된날짜입니다.
 • Y축: 예측된가격(future_predictions) - 모델이예측한비트코인가격입니다.
 • 그래프특징: 예측된가격을점선(linestyle='--')으로표시하여, 실제가격그래
프와구분할수있게합니다. 이그래프를통해미래의비트코인가격트렌드를
예측하고실제가격변동과비교할수있습니다.
 preencoded.png
 8
향후계획및발전방향
더많은데이터: 
데이터의양을늘려예측모델을향상시킬수있습니다. 또한, 다양한암호화폐
데이터를추가하여다각적인예측분석이가능하도록할예정입니다.
모델성능향상:
모델을구축할수있습니다.
 Linear Regression 외에도 머신러닝/딥러닝기법을활용하여보다정교한예측
preencoded.png
 9
결과및성과:
 • 시각화된데이터:
 • 첫번째그래프에서는비트코인가격의과거변동을시각화하여, 사용자
가가격의변화를직관적으로이해할수있도록지원합니다.
 • 두번째그래프에서는향후7일동안의비트코인가격예측을제공, 이
를통해사용자들이향후시세변동에대한예측을할수있도록합니다.
 • 사용자경험개선: 시세예측결과를한글로제공하여한국어사용자가직관
적으로데이터를분석하고예측할수있도록작업 및 개선하였습니다.
 preencoded.png
 10
읽어주셔서감사합니다.
 preencoded.png
 end
