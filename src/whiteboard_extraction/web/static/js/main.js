import { applyTreeState } from "./tree.js";
import "./regions.js";
import "./forms.js";
import "./dragdrop.js";
import "./tasks.js";
import "./upload.js";
import "./player.js";

applyTreeState();
document.addEventListener("regions:updated", applyTreeState);
