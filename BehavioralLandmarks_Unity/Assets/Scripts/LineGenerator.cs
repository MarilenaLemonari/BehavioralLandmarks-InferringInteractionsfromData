using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using Debug = UnityEngine.Debug;

public class LineGenerator : MonoBehaviour
{
    public GameObject linePrefab;
    Line activeLine;

    public Texture2D customCursor;
    public Vector2 cursorHotspot = Vector2.zero;

    private List<Line> lines = new List<Line>();

    public Animator transition;
    public KeyCode screenShotButton;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            StartCoroutine(GoBack());
        }

        if (Input.GetKeyDown(screenShotButton))
        {
            ScreenCapture.CaptureScreenshot(".\\Assets\\SavedProgress\\screenshot.png");
            Debug.Log("A screenshot was taken!");
        }

        Vector2 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);

        if (Input.GetMouseButtonDown(0))
        {
            GameObject newLine = Instantiate(linePrefab);
            activeLine = newLine.GetComponent<Line>();
            lines.Add(activeLine);
        }

        if (Input.GetMouseButtonUp(0))
        {
            activeLine = null;
        }

        if (activeLine != null)
        {
            activeLine.UpdateLine(mousePos);
        }

        if (Input.GetMouseButton(1))
        {
            Cursor.SetCursor(customCursor, cursorHotspot, CursorMode.Auto);

            // Delete lines when right-click is held down
            DeleteLineOnRightClick(mousePos);
        }

        if (Input.GetMouseButtonUp(1))
        {
            Cursor.SetCursor(null, Vector2.zero, CursorMode.Auto);
        }
    }

    private void DeleteLineOnRightClick(Vector2 mousePos)
    {
        foreach (Line line in lines)
        {
            if (line.IsPointInLineBounds(mousePos))
            {
                Destroy(line.gameObject);
                lines.Remove(line);
                break; // Exit the loop after deleting one line
            }
        }
    }

    IEnumerator GoBack()
    {
        transition.SetTrigger("Start2");

        yield return new WaitForSeconds(2.0f);

        SceneManager.LoadScene("Instructions");
    }
}
