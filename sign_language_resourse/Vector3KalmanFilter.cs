using UnityEngine;

public class Vector3KalmanFilter
{
    private KalmanFilter xFilter;
    private KalmanFilter yFilter;
    private KalmanFilter zFilter;

    public Vector3KalmanFilter(float processNoise = 0.008f, float measurementNoise = 0.03f)
    {
        xFilter = new KalmanFilter(processNoise, measurementNoise);
        yFilter = new KalmanFilter(processNoise, measurementNoise);
        zFilter = new KalmanFilter(processNoise, measurementNoise);
    }

    public Vector3 Update(Vector3 measurement)
    {
        float filteredX = xFilter.Update(measurement.x);
        float filteredY = yFilter.Update(measurement.y);
        float filteredZ = zFilter.Update(measurement.z);

        return new Vector3(filteredX, filteredY, filteredZ);
    }

    public void SetState(Vector3 state, float covariance)
    {
        xFilter.SetState(state.x, covariance);
        yFilter.SetState(state.y, covariance);
        zFilter.SetState(state.z, covariance);
    }
}