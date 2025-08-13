document.addEventListener("DOMContentLoaded", function () {
    const printBtn = document.getElementById("printBtn");

    if (printBtn) {
        printBtn.addEventListener("click", function (e) {
            e.preventDefault(); // stop # jump / form submit

            let printContents = document.getElementById("new_repo").innerHTML;
            let originalContents = document.body.innerHTML;

            document.body.innerHTML = printContents;
            window.print();
            document.body.innerHTML = originalContents;
        });
    }
});