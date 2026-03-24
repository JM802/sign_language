using UnityEngine;

[System.Serializable]
public class HandLandmarkData
{
    public int id;
    public float x;
    public float y;
    public float z;
}

[System.Serializable]
public class HandData
{
    public int hand_index;
    public string hand_type;
    public float bound_area;
    public string hand_gesture;
    public HandLandmarkData[] landmarks;
}

[System.Serializable]
public class GestureData
{
    public int hand_count;
    public HandData[] hands;
}
