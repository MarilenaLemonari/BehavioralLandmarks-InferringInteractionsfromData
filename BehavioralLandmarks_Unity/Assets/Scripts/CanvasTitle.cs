using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CanvasTitle : MonoBehaviour
{
    public float delay = 5f; // Delay in seconds before disabling the text
    private float time = 0f;

    private void Update()
    {
        time+= Time.deltaTime;
        if (time >= delay)
        {
            DisableText();
        }
    }

    private void DisableText()
    {
        GameObject delayObject = this.gameObject;
        delayObject.SetActive(false);
    }
}
