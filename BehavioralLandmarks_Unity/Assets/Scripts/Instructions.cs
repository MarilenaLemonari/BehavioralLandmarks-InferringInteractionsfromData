using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.SocialPlatforms.Impl;
using UnityEngine.UI;
using static System.Net.Mime.MediaTypeNames;

public class Instructions : MonoBehaviour
{
    public Animator transition;

    public void StartSketch()
    {
        StartCoroutine(Sketch());
    }

    IEnumerator Sketch()
    {
        transition.SetTrigger("Start");

        yield return new WaitForSeconds(1.0f);

        SceneManager.LoadScene("SampleScene");

    }
}
