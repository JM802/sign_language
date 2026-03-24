using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Hold : MonoBehaviour
{
    public GameObject target;
    public bool hold;
    private Vector3 lastPosition;
    private Quaternion lastRotation;
    private float timer;

    void Start()
    {
        lastPosition = transform.position;
        lastRotation = transform.rotation;
    }

    void Update()
    {
        if (hold) { timer = 1; }
        timer -= Time.deltaTime;

        transform.localPosition = new Vector3(0, 0, 0.08f);
        Vector3 positionDelta = transform.position - lastPosition;
        Quaternion rotationDelta = transform.rotation * Quaternion.Inverse(lastRotation);

        if (target != null && timer > 0)
        {
            target.transform.position += positionDelta;
            Vector3 pivotOffset = target.transform.position - transform.position;
            target.transform.position = transform.position + rotationDelta * pivotOffset;
            target.transform.rotation = rotationDelta * target.transform.rotation;
        }

        lastPosition = transform.position;
        lastRotation = transform.rotation;

        transform.localPosition = new Vector3(0, 0, 0.08f);
    }

    // 使用触发器事件替代碰撞事件
    void OnTriggerEnter(Collider other)
    {
        target = other.gameObject;
    }

    void OnTriggerExit(Collider other)
    {
        target = null;
    }
}