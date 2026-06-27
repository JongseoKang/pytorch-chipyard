documentation


Installation
pytorch-chipyard github repo의 readme를 그대로 따라하면됨. 해당 레포는 chipyard/firesim 까지만 포함하고 있고, vivado 를 비롯해 fpga 세팅은 포함하고 있지 않음. (구체적으로 뭘 더 설치해야하는지는 너가 이 글 옆에 적어: {TODO} )
또한 해당 프로젝트는 버전에 굉장히 민감하기 때문에, triton, chipyard commit 을 고정시켜놓았는데, 혹시나 다른 커밋 버전에서도 작동할 수도 있음. 하지만 pytorch, llvm, buddy, triton-chipyard는 이 레포에서 포함하고 있는 custom version을 그대로 사용해야함.

Tutorials
- triton-chipyard(matmul)
우리의 논문은 torch model 실행을 목적으로 했지만 부수적으로 triton-chipyard 역시 독립적으로 사용이 가능하다. Installation을 그대로 따라하면, triton-chipyard도 자동으로 같이 설치되는데, triton-chipyard를 단독으로 사용하고 싶으면, "verilator 경로 환경변수"를 설정해서 verilator를 통해 host python 안에서 bare-metal로 커널을 측정해볼 수 있다.

triton_chipyard repo 안의 example을 그대로 따라 할 수 있는데, 여기서는 가장 간단한 matmul 예제를 설명한다.(참고로 이 코드의 triton kernel 거의 triton-shared repo에서 복사해온건데, citation 달아야되나? 이거 한번 확인해봐).

```
대충 ./triton_chipyard/example/test_matmul.py의 코드
```
보여주고, 

%%%실행 명령어와 필요한 환경변수 세팅 설명.
너가 triton_chipyard 분석해서 작성해
%%%

- pytorch-chipyard(resnet50)

```
대충 ./examples/ResNet50.py 코드
```
보여주고,

%%%실행 명령어와 필요한 환경변수 세팅 설명
너가 ./scripts/env.sh 분석해서 작성해
이부분이 사실상 사용법을 설명하는데 매우 주요한 역할을 하니 매우 상세하게 설명하도록. 빠지는 환경변수가 없어야해.
%%%

pytorch-chipyard
- overview
./docs/figures/system-overview.pdf 넣어놨으니까 활용
- pytorch 쪽 수정사항
./docs/figures/chipyard-artifact-generation.pdf 넣어놨으니까 활용

기본적으로 host runtime 에 모델이 실행되는 
- triton-chipyard

Supported Features & Limitations

아래의 내용을 잘 분류 및 정렬해서 소제목 달아서 설명

triton-chipyard는 inductor lowering이 만드는 대부분의 op을 만들 수 있음.
다만 이와오는 무관하게, 기존 triton에서 지원하는 python api와는 몇가지 차이점이 존재함.
우선 triton 자체의 autotune 기능은 지원하지 않음. triton autotune은 host 에 kernel name, args, metadata 등을 캐싱하여, 동일한 입력이 들어오면, 다시커널을 컴파일 하지 않고, 
해당 shape에 대해서 최적의 커널을 바로 가져옴. 그러나 현재 architecture에서는 verilator를 통한 cycle accurate performance measure이 가능함에도 너무 느린 속도 때문에 host 쪽에서 
autotune을 아예 지원하고 있지 않음. 이에 따라 발생하는 가장 큰 차이점은, inductor 쪽에서 triton kernel을 생성할 때, pointwise(e.g. reduction, add 등의 loopwise 구조) 의 비교적 단순한 형태의 kernel에 대해서는 triton autotune 기능을 사용하지만, matmul, conv, attention 등 template kernel 에 대해서는 inductor 자체적으로 autotune cache를 관리하며 여러 커널을 만들게 됨. pytorch-chipyard는 inductor에서 생성한 operator 별 autotune kernel candidate들은 전부 runner.cpp에 반영해서 실제 cycle accurate autotune을 진행하지만, pointwise kernel에 대해서는 autotune을 진행하지 않고 있음.

또한 다른 큰 차이점은 tl.dot에 있는데, 원본 triton은 tl.dot에대해서 3차원 tensor가 입력으로 들어오면 batched matmul로 판단하고 lowering함. 하지만, triton-chipyard에서는 오직 2d matmul 형태만 tl.dot으로 받고 있음. 실제 batched matmul kernel을 생성할 때는, triton kernel level에서 별도의 batched matmul 을 작성해야하고, 실제 inductor template kernel에도 그렇게 반영되어있음.

구조적으로 inductor 뒤에 triton_chipyard backend를 추가한 방식이기 때문에
graph break 이 있는 모델들은 실행되지 않음. 이의 가장 대표적인 예가 hf model 들의 decoding step, torch op이 아닌 pure python if 를 갖다 박아넣어서 loop 을 돌리기 때문에  grpah break 이 생겨서 inductor가 여러번 호출됨. 현재 구조는 inductor 하나가 호출 될때마다 하나의 runner.cpp를 만드는 구조이기 때문에, 이러한 형태의 torch model을 지원하지 못함