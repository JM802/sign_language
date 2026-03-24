using UnityEngine;

public class HandLandmarkFilter
{
    private const int LandmarkCount = 21; // 每只手21个关键点
    private Vector3KalmanFilter[] leftFilters;
    private Vector3KalmanFilter[] rightFilters;
    private bool initialized;

    public HandLandmarkFilter(float processNoise = 0.008f, float measurementNoise = 0.03f)
    {
        leftFilters = new Vector3KalmanFilter[LandmarkCount];
        rightFilters = new Vector3KalmanFilter[LandmarkCount];

        for (int i = 0; i < LandmarkCount; i++)
        {
            leftFilters[i] = new Vector3KalmanFilter(processNoise, measurementNoise);
            rightFilters[i] = new Vector3KalmanFilter(processNoise, measurementNoise);
        }

        initialized = false;
    }

    public void FilterLeftHandLandmarks(HandLandmarkData[] leftHand)
    {
        if (leftHand == null)
            return;

        // 初始化滤波器状态
        if (!initialized && leftHand.Length >= LandmarkCount)
        {
            InitializeLeftFilters(leftHand);
            initialized = true;
        }

        // 对左手关键点进行滤波
        for (int i = 0; i < Mathf.Min(LandmarkCount, leftHand.Length); i++)
        {
            Vector3 rawPosition = new Vector3(leftHand[i].x, leftHand[i].y, leftHand[i].z * 4);
            Vector3 filteredPosition = leftFilters[i].Update(rawPosition);

            // 更新滤波后的数据
            leftHand[i].x = filteredPosition.x;
            leftHand[i].y = filteredPosition.y;
            leftHand[i].z = filteredPosition.z / 4;
        }
    }


    public void FilterRightHandLandmarks(HandLandmarkData[] rightHand)
    {
        if (rightHand == null)
            return;

        // 初始化滤波器状态
        if (!initialized && rightHand.Length >= LandmarkCount)
        {
            InitializeRightFilters(rightHand);
            initialized = true;
        }

        // 对右手关键点进行滤波
        for (int i = 0; i < Mathf.Min(LandmarkCount, rightHand.Length); i++)
        {
            Vector3 rawPosition = new Vector3(rightHand[i].x, rightHand[i].y, rightHand[i].z * 4);
            Vector3 filteredPosition = rightFilters[i].Update(rawPosition);

            // 更新滤波后的数据
            rightHand[i].x = filteredPosition.x;
            rightHand[i].y = filteredPosition.y;
            rightHand[i].z = filteredPosition.z / 4;
        }
    }

    private void InitializeLeftFilters(HandLandmarkData[] leftHand)
    {
        for (int i = 0; i < LandmarkCount; i++)
        {
            Vector3 leftPos = new Vector3(leftHand[i].x, leftHand[i].y, leftHand[i].z * 4);
            leftFilters[i].SetState(leftPos, 1.0f);
        }
    }

    private void InitializeRightFilters(HandLandmarkData[] rightHand)
    {
        for (int i = 0; i < LandmarkCount; i++)
        {
            Vector3 rightPos = new Vector3(rightHand[i].x, rightHand[i].y, rightHand[i].z * 4);
            rightFilters[i].SetState(rightPos, 1.0f);
        }
    }
}