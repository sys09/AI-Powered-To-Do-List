// Get elements from DOM
const taskFormEl = document.getElementById("add-task-form");
const taskListEl = document.getElementById("task-list");
const prioritizeBtn = document.getElementById("prioritize-btn");

let tasks = JSON.parse(localStorage.getItem("tasks")) || [];

// Function to add a task
function addTask(event) {
    event.preventDefault();
    const description = document.getElementById("description").value;
    const deadline = document.getElementById("deadline").value;
    const importance = parseInt(document.getElementById("importance").value);

    const task = { description, deadline, importance, priority: 0 };
    tasks.push(task);

    localStorage.setItem("tasks", JSON.stringify(tasks));
    taskFormEl.reset();
    renderTasks();
}

// Function to render tasks
function renderTasks() {
    taskListEl.innerHTML = '';
    tasks.forEach((task, index) => {
        const li = document.createElement("li");
        li.classList.add(task.priority > 0.5 ? "high-priority" : "low-priority");
        li.innerHTML = `
            ${task.description} - Deadline: ${task.deadline} - Importance: ${task.importance}
            <button onclick="removeTask(${index})">X</button>
        `;
        taskListEl.appendChild(li);
    });
}

// Function to remove a task
function removeTask(index) {
    tasks.splice(index, 1);
    localStorage.setItem("tasks", JSON.stringify(tasks));
    renderTasks();
}

// Function to prioritize tasks using AI
async function prioritizeTasks() {
    // TensorFlow.js model setup
    const model = await tf.sequential();
    
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [2] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    // Train the model with synthetic data (importance, days until deadline -> priority)
    const xs = tf.tensor2d(tasks.map(task => [task.importance, daysUntilDeadline(task.deadline)]));
    const ys = tf.tensor2d(tasks.map(task => [task.importance >= 3 ? 1 : 0])); // Simple importance threshold for training
    
    await model.fit(xs, ys, { epochs: 10 });

    // Make predictions for each task
    const predictions = model.predict(xs).dataSync();
    
    tasks = tasks.map((task, i) => ({
        ...task,
        priority: predictions[i]
    }));

    localStorage.setItem("tasks", JSON.stringify(tasks));
    renderTasks();
}

// Helper function to calculate days until deadline
function daysUntilDeadline(deadline) {
    const today = new Date();
    const deadlineDate = new Date(deadline);
    const timeDiff = deadlineDate.getTime() - today.getTime();
    return Math.ceil(timeDiff / (1000 * 3600 * 24));
}

// Event listeners
taskFormEl.addEventListener("submit", addTask);
prioritizeBtn.addEventListener("click", prioritizeTasks);

// Initialize the app
renderTasks();
