# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 신재현
- 리뷰어 : 최호재


🔑 **PRT(Peer Review Template)**

- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)** (3/3)
    > 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    > 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 퀘스트 문제 요구조건 등을 지칭
    > 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
    - [x] CAM을 얻기 위한 기본모델의 구성과 학습이 정상 진행되었는가?
      - ResNet50 + GAP + DenseLayer 결합된 CAM 모델의 학습과정이 안정적으로 수렴하였다.![image](https://github.com/HectorSin/Aiffel/assets/98305832/4d262fd9-2cae-4351-b35f-5f56c47838a4)
    - [x] 분류근거를 설명 가능한 Class activation map을 얻을 수 있는가?
      - CAM 방식과 Grad-CAM 방식의 class activation map이 정상적으로 얻어지며, 시각화하였을 때 해당 object의 주요 특징 위치를 잘 반영한다.
        - CAM: ![image](https://github.com/HectorSin/Aiffel/assets/98305832/ae0c3b3f-1b13-4400-927a-7d2a1bc626d4)
        - Grad-CAM: ![image](https://github.com/HectorSin/Aiffel/assets/98305832/e3114507-0ea0-4e92-917e-e2af2bbb3d27)
    - [x] 인식결과의 시각화 및 성능 분석을 적절히 수행하였는가?
        - CAM과 Grad-CAM 각각에 대해 다음의 과정을 통해 CAM과 Grad-CAM의 object localization 성능이 비교분석되었다.
            - 원본이미지합성![image](https://github.com/HectorSin/Aiffel/assets/98305832/86128d29-c44f-4e48-b8fa-c969c29ad1eb),
            - 바운딩박스 ![image](https://github.com/HectorSin/Aiffel/assets/98305832/ac346ffb-162e-4d86-a14e-389c49935e10)
            - IoU 계산 ![image](https://github.com/HectorSin/Aiffel/assets/98305832/a65c6024-544d-4cde-bd50-e776e7ca946a)

- [x]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)** (2/3)
    - [ ]  모델 선정 이유
    - [x]  CAM, Grad-CAM 결과를 가독성 높은 배치, 설명으로 시각화 하였다. ![image](https://github.com/HectorSin/Aiffel/assets/98305832/23e7241f-8368-4750-bb92-18f8e93575c6) ![image](https://github.com/HectorSin/Aiffel/assets/98305832/334021c0-3324-42bf-bd5f-31823a3cb724)


    - [x]  CAM, Grad-CAM 생성 함수 구현 방안
        - ![image](https://github.com/HectorSin/Aiffel/assets/98305832/03efedf0-62ff-4216-bdb2-683ed6ac71c5)
        - generate_grad_cam 함수에 각 부분에 대한 설명을 주석으로 적음 ![image](https://github.com/HectorSin/Aiffel/assets/98305832/0a0697ef-d622-4382-8506-65e5285e01e7)


- [x]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)** (4/4)
    - [x]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)![image](https://github.com/HectorSin/Aiffel/assets/98305832/69144c5f-8df4-4370-931a-2499d5940bd2)
    - [x]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)![image](https://github.com/HectorSin/Aiffel/assets/98305832/f999d12d-1d46-4c05-8cc6-37872b3a674c)
    - [x]  각 실험을 시각화하여 비교하였나요? ![image](https://github.com/HectorSin/Aiffel/assets/98305832/bfaf475e-8c1f-4c96-8498-dff7e93521f6)
    - [x]  모든 실험 결과가 기록되었나요?

- [x]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)** (4/4)
    - [image](https://github.com/HectorSin/Aiffel/assets/98305832/63faa6e1-27bd-4866-bb36-a10ae0e6cd9d)
    - [x]  배운 점
    - [x]  아쉬운 점
    - [x]  느낀 점
    - [x]  어려웠던 점
