# visualizer.py
import webbrowser
import os

def create_html_report(results):
    """
    Generates an HTML file highlighting the bias.
    results: List of dicts -> [{'text': str, 'biases': [(start, end, word), ...]}, ...]
    """
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; }
            .card { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
            .bias { background-color: #ffcccc; color: #cc0000; font-weight: bold; padding: 2px 4px; border-radius: 4px; }
            .safe { background-color: #ccffcc; color: #006600; padding: 2px 4px; border-radius: 4px; }
            h2 { color: #333; }
            .legend { margin-bottom: 20px; padding: 10px; background: #f9f9f9; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Pronoun Bias Detection Report</h1>
        <div class="legend">
            <span class="bias">Red Highlight</span> = Potential Generic Bias <br>
            <span class="safe">Green Highlight</span> = Specific/Anchored (Safe)
        </div>
    """

    for i, res in enumerate(results):
        text = res['text']
        biases = res['biases']
        
        # Sort spans by start index
        biases.sort(key=lambda x: x[0])
        
        formatted_text = ""
        cursor = 0
        
        for start, end, word in biases:
            # Append text before the bias
            formatted_text += text[cursor:start]
            # Append the biased word with highlighting
            formatted_text += f'<span class="bias" title="Generic Context">{text[start:end]}</span>'
            cursor = end
            
        # Append remaining text
        formatted_text += text[cursor:]
        
        status = "⚠️ BIAS DETECTED" if biases else "✅ SAFE"
        color = "red" if biases else "green"
        
        html_content += f"""
        <div class="card" style="border-left: 5px solid {color};">
            <h3>Document {i+1}: {status}</h3>
            <p>{formatted_text.replace(chr(10), '<br>')}</p>
        </div>
        """

    html_content += "</body></html>"
    
    # Save and Open
    filename = "bias_report.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\nReport generated: {os.path.abspath(filename)}")
    webbrowser.open(f"file://{os.path.abspath(filename)}")