document.addEventListener("DOMContentLoaded", function() {
    // Find the "Generate Speech Draft" button by its text content.
    const genButton = Array.from(document.querySelectorAll("button"))
      .find(btn => btn.textContent.trim() === "Generate Speech Draft");
      
    if (genButton) {
      genButton.addEventListener("click", function(e) {
        e.preventDefault();
        generateSpeechDraft();
      });
    }
  });
  
  function generateSpeechDraft() {
    // Retrieve saved data from page1, page2, and page3.
    const page1Data = JSON.parse(localStorage.getItem("page1Data") || "{}");
    const page2Data = JSON.parse(localStorage.getItem("page2Data") || "{}");
    const page3Data = JSON.parse(localStorage.getItem("page3Data") || "{}");
    
    // Combine all data into one object.
    const combinedData = Object.assign({}, page1Data, page2Data, page3Data, { page: "page3" });
    
    // Send the combined data to the Flask server's /process endpoint (JWT required).
    const token = (typeof getAuthToken === "function" ? getAuthToken() : null)
      || localStorage.getItem("authToken");
    if (!token) {
      alert("Please log in first.");
      window.location.href = "/login";
      return;
    }
    fetch('/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
      },
      body: JSON.stringify(combinedData)
    })
    .then(response => response.json())
    .then(result => {
        const speechText = result.response; // Get text from backend
        // Extract speech (until "key_themes")
          const speechMatch = speechText.match(/"speech":\s*"([\s\S]*?)"\s*,\s*"key_themes"/);
          const speech = speechMatch ? speechMatch[1].trim() : "";
  
          // Extract key themes
          const keyThemesMatch = speechText.match(/"key_themes":\s*\[\s*([\s\S]*?)\s*\]/);
          const keyThemes = keyThemesMatch 
              ? keyThemesMatch[1].match(/"([^"]+)"/g).map(s => s.replace(/"/g, '')).join(', ') 
              : "";
  
          // Extract category
          const categoryMatch = speechText.match(/"category":\s*"([^"]+)"/);
          const category = categoryMatch ? categoryMatch[1] : "";
  
          // Extract explanation (until end)
          const explanationMatch = speechText.match(/"explanation":\s*"([\s\S]*?)"\s*\}/);
          const explanation = explanationMatch ? explanationMatch[1].trim() : "";
  
          console.log("Speech:", speech);
          console.log("Key Themes:", keyThemes);
          console.log("Category:", category);
          console.log("Explanation:", explanation);
          document.getElementById("speech-draft").value = speech; // Display in textarea
    })
    .catch(error => {
      console.error("Error generating speech draft:", error);
      document.getElementById("speech-draft").value = "Error generating speech draft.";
    });
  }
  