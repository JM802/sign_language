### 一 、初始维数怎么计算的

1. MediaPipe Holistic 模型标准的 Pose 检测包含 33 个关键点。
2. 但是本项目主要做的手语（上半身），0-24点包含全部上半身
3. mp标准输出左右手分别21个点
4. x,y计算逻辑：mp提供了深度z信息，但是手语一般仅需要2维，即x，y坐标，降低模型复杂度，便于训练

### 二、视频预处理流程（extract_features）

1. 通过start_frame,end_frame指定帧区间
2. 通过for循环限制0-25，只选取pose的上半身，其余选择左右手全部21点，未识别到的用（0，0）填充
3. 选取动态视频模式，**基于视频的帧间关联性**来跟踪关键点
4. lm_x,lm_y指相对画面（归一化）的坐标，避免因为画面大小导致识别错误

### 三、参数的标准化

1. 通过正态分布公式对每一列即每一个手语特征做标准化

### 四、原始json数据流转

1. json格式：{"video_001": {"subset":"train", "action":[0,10,50]}}，标明video_id,subset_name(train.val,test),label,start_frame,end_frame
2. extract_features函数处理生成video_id.npy文件，shape为有效帧*134（用for循环遍历），每一个视频对应一个npy,用video_id区分
3. 按subset类型分类，subsets[subset]存储{npy_save_path},{label}，比如上面的例子train列表里面是npy_save_path,label
4. {subset_name}_map_300.txt 这个map_file里面每一行存储npy_save_path,label,有三个train,val,test
5. 训练集归一化，测试 / 验证集也用训练集的 mean/std，避免数据泄露
