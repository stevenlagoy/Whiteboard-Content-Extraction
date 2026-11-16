// Timestamps beside recognized text jump the video to that moment.

document.addEventListener("click", (event) => {
  const button = event.target.closest?.("[data-seek]");
  const player = document.querySelector("[data-player]");
  if (!button || !player) return;
  player.currentTime = Number(button.dataset.seek);
  player.play().catch(() => {});
});
