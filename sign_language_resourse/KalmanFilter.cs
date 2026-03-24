using UnityEngine;

public class KalmanFilter
{
    private float q; // 过程噪声协方差
    private float r; // 测量噪声协方差
    private float x; // 估计值
    private float p; // 估计误差协方差
    private float k; // 卡尔曼增益

    public KalmanFilter(float processNoise, float measurementNoise)
    {
        q = processNoise;
        r = measurementNoise;
        p = 1.0f;
    }

    public float Update(float measurement)
    {
        // 预测
        p = p + q;

        // 计算卡尔曼增益
        k = p / (p + r);

        // 更新估计值
        x = x + k * (measurement - x);

        // 更新估计误差协方差
        p = (1 - k) * p;

        return x;
    }

    public void SetState(float state, float covariance)
    {
        x = state;
        p = covariance;
    }
}