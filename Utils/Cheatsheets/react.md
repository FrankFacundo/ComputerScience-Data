# React

Basic Hooks

- useState
- useEffect
- useContext

Additional Hooks

- useCallback

## Basic Hooks

### useState

The useState hook is a fundamental hook used for state management in functional components. It lets you add React state to functional components. When you call this hook, it returns an array containing two elements: the current state value and a function that lets you update it. You can use this hook to keep the component's UI and state in sync.

```js
import React, { useState } from "react";

function Counter() {
  // Declare a new state variable, which we'll call "count"
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>Click me</button>
    </div>
  );
}
```

---

### useEffect

The useEffect hook is one of the essential hooks provided by React, allowing functional components to perform side effects.
The broader point of useEffect is to provide a unified and declarative interface for handling side effects in functional components. Before hooks, class components used lifecycle methods like componentDidMount, componentDidUpdate, and componentWillUnmount to execute side effects. However, these lifecycle methods often led to scattered and repetitive code since related logic could be spread across different methods.

Example 1: Fetching Data

```js
useEffect(() => {
  const fetchData = async () => {
    const response = await fetch("https://api.example.com/data");
    const data = await response.json();
    // Assume setState is a state setter function from useState
    setState(data);
  };

  fetchData();
}, []); // Empty array means this runs once after the initial render
```

In this example, `useEffect` is used to fetch data from an API and update the state accordingly. The empty dependency array ensures that the effect runs only once after the component mounts, similar to `componentDidMount` in class components.

Example 2: Adding and Removing Event Listeners

```js
useEffect(() => {
  const handleResize = () => {
    // Handle the resize action
  };

  window.addEventListener("resize", handleResize);

  // Cleanup function
  return () => {
    window.removeEventListener("resize", handleResize);
  };
}, []); // Empty array means this effect is only applied once
```

Here, `useEffect` is used to add an event listener to the window object when the component mounts and remove it when the component unmounts. The cleanup function ensures that we clean up the event listener to avoid memory leaks.

Example 3: Updating Document Title

```js
useEffect(() => {
  document.title = `You clicked ${count} times`;
}, [count]); // Runs whenever `count` changes
```

This example demonstrates using `useEffect` to update the document title every time the `count` state changes. The dependency array contains `count`, so the effect re-runs whenever `count` is updated.

---

### useContext

The `useContext` hook is a part of React's Context API, allowing you to share values like props, state, and functions across your entire application or part of it, without having to explicitly pass them through each level of the component tree. It's particularly useful for managing global state, themes, authentication, and more.

#### Basic Concept of Context

Before diving into `useContext`, it's crucial to understand the concept of Context in React. Context provides a way to pass data through the component tree without having to pass props down manually at every level. In a typical React application, data is passed top-down (parent to child) via props, but this can become cumbersome for certain types of props (like locale preference, UI theme) that are required by many components within an application. Context is designed to share values like these across all levels of the application without the prop-drilling.

#### Creating a Context

To use `useContext`, you first need to create a Context using `React.createContext()`. This returns an object with two components, `Provider` and `Consumer`, though with hooks, you typically only use the `Provider` and `useContext`.

**Example: Creating a Context for a theme**

```javascript
import React from "react";

// Create a Context
const ThemeContext = React.createContext();
```

#### Using `Provider` to Pass the Context

The `Provider` component is used to wrap a part of your application where you want the context to be accessible. It has a `value` prop to pass down the context to the components.

```javascript
<ThemeContext.Provider value={{ theme: "dark" }}>
  // Your component tree that needs access to this context
</ThemeContext.Provider>
```

#### Accessing Context with `useContext`

`useContext` allows you to subscribe to React context within functional components. The hook accepts the context object (the value returned from `React.createContext`) and returns the current context value, which is the `value` prop of the nearest `<Context.Provider>` above in the component tree.

**Example: Accessing the theme context**

```javascript
import React, { useContext } from "react";

const ThemedComponent = () => {
  // Accessing context value
  const themeContext = useContext(ThemeContext);

  return (
    <div
      style={{
        backgroundColor: themeContext.theme === "dark" ? "black" : "white",
      }}
    >
      The theme is {themeContext.theme}
    </div>
  );
};
```

#### Full Example

Putting it all together:

```javascript
import React, { useContext, createContext } from "react";

// Step 1: Create a context
const ThemeContext = createContext();

const App = () => {
  return (
    // Step 2: Provide a context value
    <ThemeContext.Provider value={{ theme: "dark" }}>
      <Toolbar />
    </ThemeContext.Provider>
  );
};

const Toolbar = () => {
  return (
    <div>
      <ThemedButton />
    </div>
  );
};

const ThemedButton = () => {
  // Step 3: Consume the context value
  const { theme } = useContext(ThemeContext);

  return (
    <button
      style={{
        background: theme === "dark" ? "black" : "white",
        color: theme === "dark" ? "white" : "black",
      }}
    >
      I am styled by theme context!
    </button>
  );
};

export default App;
```

In this example, `ThemeContext.Provider` wraps the `Toolbar` component, making the `theme` context available to any component in the app, including `ThemedButton`. `ThemedButton` uses `useContext` to access the `theme` and apply styling accordingly.

#### Benefits of `useContext`

- **Avoids Prop Drilling**: Directly pass data to the component that needs it without passing through intermediate components.
- **Simplifies Component API**: Reduces the number of props that need to be passed around by storing them in context.
- **Performance Optimization**: React ensures that components consuming the context only re-render when the context value changes.

`useContext` simplifies the way you manage and access state across your application, making your code cleaner and more maintainable.

---

## Additional Hooks

### useCallback

The useCallback hook is used to memoize callbacks. This means that the function is not recreated on every render unless one of its dependencies changes.This is useful for optimizing performance, especially for components that rely on reference equality to prevent unnecessary renders.

```js
import React, { useCallback, useState } from "react";

function Example() {
  const [count, setCount] = useState(0);

  const increment = useCallback(() => {
    setCount((c) => c + 1);
  }, [setCount]); // Dependencies array

  return (
    <div>
      Count: {count}
      <button onClick={increment}>+</button>
    </div>
  );
}
```

In this example, useCallback ensures that the increment function is not recreated on every render unless setCount changes, which it realistically doesn't in this context. This optimization can be beneficial in more complex components where prop comparisons or expensive operations are involved.

## References:

- https://legacy.reactjs.org/docs/hooks-reference.html
- https://react.dev/reference/react/hooks
