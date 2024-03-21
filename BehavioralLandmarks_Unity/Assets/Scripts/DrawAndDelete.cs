using UnityEngine;

public class DrawAndDelete : MonoBehaviour
{
    public GameObject drawingPrefab; // Reference to the object you want to draw

    void Update()
    {
        if (Input.GetMouseButtonDown(0)) // Left mouse button to draw
        {
            Vector3 mousePosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            mousePosition.z = 0;
            Instantiate(drawingPrefab, mousePosition, Quaternion.identity);
        }

        if (Input.GetMouseButtonDown(1)) // Right mouse button to delete
        {
            Vector3 mousePosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            mousePosition.z = 0;

            Collider2D[] colliders = Physics2D.OverlapCircleAll(mousePosition, 0.1f);
            foreach (Collider2D collider in colliders)
            {
                if (collider.gameObject == gameObject)
                {
                    Destroy(collider.gameObject);
                }
            }
        }
    }
}
