using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class PointDrawer : MonoBehaviour
{
    public GameObject pointPrefab; // Reference to the point prefab
    public float minScale = 1f;
    public float maxScale = 2f; // Maximum scale for the point
    public float maxSpeed = 10f;

    private bool isDrawing = false; // Flag to track if drawing is in progress
    private Vector3 lastMousePosition; // Store the last mouse position for speed calculation
    private float timeSinceLastPoint = 0f; // Time elapsed since the last point was created
    public float minTime = 0.1f; // Minimum time between points

    private bool isFirst = true;

    public Texture2D customCursor;
    public Vector2 cursorHotspot = Vector2.zero;
    public Animator transition;
    public KeyCode screenShotButton;
    public InputField areaSize;
    public string userName = "Mlemonari";

    private List<Vector2> dotPositions = new List<Vector2>(); //new
    private List<float> recordedSpeeds = new List<float>();
    private List<float> timeStamps = new List<float>();
    private int dotCount = 0;
    private float currentTime;
    private Dictionary<Vector2, float> timeDictionary = new Dictionary<Vector2, float>();

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            StartCoroutine(GoBack());
        }

        if (Input.GetKeyDown(screenShotButton))
        {
            areaSize.gameObject.SetActive(true);
            areaSize.Select();
            areaSize.ActivateInputField();  
        }

        if (Input.GetKeyDown(KeyCode.Return))
        {
            string areaDim = areaSize.text;
            areaSize.gameObject.SetActive(false);
            ScreenCapture.CaptureScreenshot($".\\Assets\\SavedProgress\\.Crowd_{userName}_{areaDim}.tif");
            Debug.Log("A screenshot was taken!");
        }

        // Check if the left mouse button is held down
        if (Input.GetMouseButton(0))
        {
            // Start drawing or continue drawing
            DrawPoint();
        }
        else if (Input.GetMouseButtonUp(0))
        {
            // Stop drawing when the left mouse button is released
            isDrawing = false;
            currentTime = 0;
            SaveDotPositionsToCSV(); //new
        }

        // Check if the right mouse button is held down
        if (Input.GetMouseButton(1))
        {
            Cursor.SetCursor(customCursor, cursorHotspot, CursorMode.Auto);

            // Erase points at the current mouse position
            ErasePoints();
        }

        if (Input.GetMouseButtonUp(1))
        {
            Cursor.SetCursor(null, Vector2.zero, CursorMode.Auto);
        }
    }

    float ExponentialDecay(float x, float maxValue, float decayRate)
    {
        return maxValue * Mathf.Exp(-decayRate * x);
    }

    private void DrawPoint()
    {
        // Check if drawing is not in progress, and if so, set the flag and create the first point
        if (!isDrawing)
        {
            isDrawing = true;
            CreatePoint();
        }

        // Get the mouse position in world coordinates inside the Update method
        Vector3 mousePosition = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, Camera.main.transform.position.z * -1));
        mousePosition.z = 0; // Ensure the Z position is set to 0

        // Calculate the time since the last point was created
        timeSinceLastPoint += Time.deltaTime;

        // Check if the minimum time threshold is met or if the speed is above a certain threshold
        if (timeSinceLastPoint >= minTime)
        {
            if (isFirst == true)
            {
                isFirst = false;
                lastMousePosition = mousePosition;
            }

            // Calculate the speed based on the distance between the last mouse position and the current mouse position
            float speed = Vector3.Distance(mousePosition, lastMousePosition) / timeSinceLastPoint;
            currentTime += minTime;

            // Calculate the scale based on speed
            // float scale = ExponentialDecay(speed, maxScale, (-Mathf.Log(1 / maxScale) / maxSpeed));
            float scale = Mathf.Lerp(maxScale, minScale, Mathf.InverseLerp(0f, maxSpeed, speed));

            // Create a new point at the mouse position with the calculated scale
            // Debug.Log("Log " + mousePosition.x + " & " + mousePosition.y + " & " + speed);
            CreatePoint(mousePosition, scale, speed, currentTime);

            // Reset the timer
            timeSinceLastPoint = 0f;

            // Update the last mouse position for the next frame
            lastMousePosition = mousePosition;
        }
    }

    private void CreatePoint(Vector3 position = default, float scale = 1f, float currentSpeed = 0f, float currentTime = 0f)
    {
        // Instantiate a point GameObject at the specified position or at the origin
        GameObject point = Instantiate(pointPrefab, position, Quaternion.identity);

        // Set the scale of the point based on the calculated scale
        point.transform.localScale = new Vector3(scale, scale, 1f);

        // Add the position to the list
        dotPositions.Add(new Vector2(position.x, position.y)); //new
        recordedSpeeds.Add(currentSpeed);
        timeStamps.Add(currentTime);
    }

    private void ErasePoints()
    {
        // Get the mouse position in world coordinates
        Vector3 mousePosition = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, Camera.main.transform.position.z * -1));
        mousePosition.z = 0; // Ensure the Z position is set to 0

        // Detect and erase points at the current mouse position
        Collider2D[] colliders = Physics2D.OverlapCircleAll(mousePosition, 0.1f);

        foreach (Collider2D collider in colliders)
        {
            if (collider.CompareTag("Point"))
            {
                // Destroy the point GameObject
                Destroy(collider.gameObject);
            }
        }
    }

    IEnumerator GoBack()
    {
        transition.SetTrigger("Start2");

        yield return new WaitForSeconds(2.0f);

        SceneManager.LoadScene("Instructions");
    }

    //new
    private void SaveDotPositionsToCSV()
    {
        // Create a unique file name based on a timestamp
        string fileName = $"{userName}_{areaSize.text}_dots_{dotCount}.csv";

        // Define the path where you want to save the CSV file
        string filePath = Application.dataPath + $"/SavedDots/.{fileName}";

        // Create a StringBuilder to build the CSV content
        System.Text.StringBuilder csvContent = new System.Text.StringBuilder();

        //// Add headers to the CSV
        //csvContent.AppendLine("X,Y");

        // Add the dot positions to the CSV
        int index = 0;
        foreach (Vector2 position in dotPositions)
        {
            csvContent.AppendLine($"{position.x},{position.y},{recordedSpeeds[index]}");
            index++;
        }

        // Write the CSV content to the file
        System.IO.File.WriteAllText(filePath, csvContent.ToString());

        // Reset the dot positions list and increment the dot count
        dotPositions.Clear();
        dotCount++;
        recordedSpeeds.Clear();
    }
}
