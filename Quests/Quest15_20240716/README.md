### 07.17 코드 리뷰
***
### 코더 : 신재현님
### 리뷰어 : 고대현
***
### 리뷰 총평

저는 실험을 할 때 Augmentation에 대해서 깊게 실험을 진행해보지 못했는데, 재현님께서는 Augementation을 했을 때의 문제점 및 개선 방향을 잘 작성해주셔서 실험을 더 깊이 이해할 수 있게 되었습니다.

또한 U-Net 모델과 U-Net++ 모델을 잘 구현해주시고, 실험 결과 또한 굉장히 잘 되어서 좋은 고민 뒤에는 좋은 결과가 따른다는 생각을 하게 되었습니다.

재현님, 실험 및 결과까지 내시느라 고생하셨습니다 :D
***
🔑 **PRT(Peer Review Template)**

- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
> 문제에서 요구하는 최종 결과물이 첨부되었고, 루브릭 모두를 충족하였습니다!
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
    - 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
     
    ![image](https://github.com/user-attachments/assets/25ac8349-47cc-4f74-b69f-c750b213f866)


- [X]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
 > 모델 선정 이유에 대해서 잘 나와 있었습니다!
    - [X]  모델 선정 이유
    - [ ]  Metrics 선정 이유
    - [ ]  Loss 선정 이유

- [X]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - [X]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
      > 데이터를 분할하여 프로젝트를 진행하셨고, 훈련용 데이터인 경우에는 따로 augementation을 수행해주셨습니다.
      > ![image](https://github.com/user-attachments/assets/e77e7798-04d0-4f71-8ffe-dbffac09e882)
    - [X]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
    - [X]  각 실험을 시각화하여 비교하였나요?
      > 각 실험 결과를 시각화하여 비교해주셨고, 특히 원인 분석에서 인상 깊었습니다.
      > ![image](https://github.com/user-attachments/assets/b7927e75-585b-4e13-a69f-536953b650f4)
    - [X]  모든 실험 결과가 기록되었나요?
      > 모든 실험 결과가 기록되었고, 원인 분석 - 가설 - 검증의 순서대로 실험을 진행해주시고, 결과까지 보셨습니다.
      > ![image](https://github.com/user-attachments/assets/fa8eb0d0-e1f4-4e64-92ef-b2df161fa64f)
      > ![image](https://github.com/user-attachments/assets/1ab5db99-f634-4043-9a2f-35ee3aafb555)

- [ ]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [ ]  배운 점
    - [ ]  아쉬운 점
    - [ ]  느낀 점
    - [ ]  어려웠던 점